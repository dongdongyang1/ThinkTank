import re
from typing import List, Dict, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.logger import logger
from processor.query_processor.prompt.answer_out import ANSWER_PROMPT
from processor.query_processor.state import QueryGraphState
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_messages, get_recent_messages
from utils.sse_utils import SSEEvent
from utils.task_utils import push_to_session, set_task_result, push_to_session_nowait

MAX_CONTEXT_CHARS = 12000

class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案输出
    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """
    # 覆盖基类的 name 属性，标识节点名称
    name :str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
            1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
            2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
            3 调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
            4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
            5 做最后一次push操作（主要是为了触发前端图片渲染)
                {
                    "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
                    "status": "completed",
                    "image_urls": [
                        "http://local-server/images/panel_view.jpg",
                        "http://local-server/images/button_detail.jpg"
                    ]
                }
        """

        logger.info(f"【{self.name}】节点逻辑")
        # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
        answer_exists = self._step_1_check_answer(state)

        # 阶段二  如果没有answer则 构建 Prompt
        if not answer_exists:
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt

            # 阶段三：  如果没有answer则 调用大模型输出答案
            self._step_3_generate_response(state,prompt)

        # 提取图片URL（用于历史记录和前端展示）
        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])

        # 非流式模式下，把image_urls存入task_result，供/query接口返回
        if not state.get("is_stream"):
            set_task_result(state["session_id"], "image_urls", image_urls)

        # 阶段四：把答案写入到mongodb的history中
        if state.get("answer"):
            logger.info("---写入MongoDB历史记录---")
            self._step_4_write_history(state,image_urls=image_urls)

        # 阶段五: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能争取渲染和结束]
        logger.info(f"---发送 final 事件---图片为：{image_urls}")

        push_to_session_nowait(
            state["session_id"],
            SSEEvent.FINAL,
            {
                "answer":state["answer"],
                "status": "completed",
                "image_urls": image_urls  # 发送图片URL给前端
            }
        )

        logger.info("---node_answer_output 节点处理结束---")
        return state

    def _step_1_check_answer(self, state)->bool:
        """
        阶段一：检查 state 中是否已有 answer。
        - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
        - 若不存在：返回 False
        """
        answer = state.get("answer")
        is_stream = state.get("is_stream")
        if answer:
            if is_stream:
                logger.info("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session_nowait(state["session_id"],SSEEvent.DELTA,{"delta":answer})
            else:
                set_task_result(state["session_id"],"answer",answer)
            return True
        else:
            return False

    def _step_2_construct_prompt(self, state:QueryGraphState)->str:
        """
           阶段二：构建 Prompt
           根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组装 LLM 提示词
        """
        char_budget = MAX_CONTEXT_CHARS

        # 1. 获取问题和商品名
        # 优先使用重写后的问题
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 2. 格式化上下文文档
        context_str,char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化 Item Names (提问商品)
        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        # 5. 组装提示词
        prompt = ANSWER_PROMPT.format(
            context = context_str or "无参考内容",
            history = history_str if history_str else "暂无历史对话",
            item_names = item_names_str,
            question = question
        )
        logger.info(f"组装后的提示词为：{prompt}")
        return prompt

    def _format_reranked_docs(self, reranked_docs:List[Dict], char_budget:int) ->Tuple[str,int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0
        for idx,doc in enumerate(reranked_docs,start=1):
            content = doc.get("content") or ""
            meta_tags = [f"[{idx}]"]
            for field,template in [
                ("source", "[source={}]"),
                ("chunks_id", "[chunks_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))
            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")
            doc_entry = "".join(meta_tags)+"\n"+content

            if used_chars + len(doc_entry) > char_budget:
                break
            formatted_lines.append(doc_entry)
            used_chars+=len(doc_entry) + 2

        return "\n\n".join(formatted_lines),char_budget-used_chars

    def _format_chat_history(self, chat_history:List[Dict], char_budget:int)->Tuple[str,int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {
            "user":"用户","assistant":"助手"
        }
        for message in chat_history:
            role = message.get("role","")
            text = message.get("text","")
            if not text or role not in role_label_map:
                continue
            formatted_line = f"{role_label_map[role]}:{text}"
            line_len = len(formatted_line)+1
            if used_chars + line_len > char_budget:
                break
            formatted_lines.append(formatted_line)
            used_chars += line_len

        return "\n".join(formatted_lines),char_budget-used_chars

    def _step_3_generate_response(self, state:QueryGraphState, prompt:str)->QueryGraphState:
        """
            阶段三：生成回答
            调用llm生成答案，支持流式输出
        """
        logger.info("---Step 3: 开始生成回答 (LLM Generation)---")

        # 获取 LLM 客户端
        # 使用统一的 get_llm_client 获取实例

        llm = get_llm_client()

        # 判断是否需要流式输出
        # 通常 state 中会注入 stream_queue 用于 SSE 推送
        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        if is_stream:
            logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk,"content","") or ""
                    if delta:
                        final_text +=delta
                        # 将增量内容放入队列
                        push_to_session_nowait(session_id,SSEEvent.DELTA,{"delta":delta})
                logger.info(f"流式输出完成，总长度: {len(final_text)}")

            except Exception as e:
                logger.error(f"流式生成出错: {e}", exc_info=True)
                # 发生错误时，尝试推送到前端
                push_to_session_nowait(session_id, SSEEvent.ERROR, {"error": str(e)})

            state["answer"] = final_text
        else:
            # 非流式直接调用
            logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id,"answer",content)
                logger.info(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                if not prompt:
                    logger.error(f"LLM 生成失败: {e}", exc_info=True)
                    state["answer"] = ""

        return state

    def _extract_images_from_docs(self, docs):
        """
            辅助方法：从文档列表中提取图片URL

            核心逻辑：
            1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
            2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
               - 验证后缀名是否为图片格式 (.jpg, .png 等)。
            3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
               - 匹配 Markdown 图片语法: ![alt text](image_url)。
            4. 对提取到的 URL 进行去重处理，返回唯一图片列表。

            :param docs: 文档列表，每个文档为字典格式
            :return: 图片 URL 字符串列表
        """
        images = []
        seen = set()  # 用于去重，避免同一张图片重复出现
        if not docs:
            return []

        md_img_pattern = re.compile(r'!\[.*?\]\((.*?\.(?:jpg|jpeg|png|gif|webp|bmp|svg))\)', re.IGNORECASE)
        logger.info(f"开始提取图片，待处理文档数: {len(docs)}")
        for i,doc in enumerate(docs):
            # 1. 优先检查 url 字段 (主要针对 Web Search 结果)
            url = (doc.get("url") or "").strip()
            if url:
                # 简单后缀判断：确保是静态图片资源
                if url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                        seen.add(url)
                        images.append(url)

             # 2. 检查 text 字段中的 Markdown 图片 (主要针对 Local Chunk)
            text = (doc.get("content") or "").strip()
            if text:
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    if img_url and img_url not in seen:
                        logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                        seen.add(img_url)
                        images.append(img_url)

        logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
        return images

    def _step_4_write_history(self, state:QueryGraphState, image_urls = None) ->QueryGraphState:
        """
        阶段四：把本轮答案写入 MongoDB history。
        利用 utils/mongo_history_utils.py 中的 save_chat_messages 方法。
        """
        session_id = state.get("session_id","default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []
        try:
            if answer:
                save_chat_messages(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query=state.get("rewritten_query", ""),
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
                logger.info("MongoDB写入记录成功")
        except Exception as e:
            # 写历史失败不应影响主链路
            logger.error(f"写入Mongo历史记录失败: {e}",exc_info=True)

        return state


if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "content": "烫金机 可选7PPM烫金速度  无版烫印  配备最大44页标准ADF进纸器  支持省膜模式  10字符x2行LCD液晶屏  HAK180烫金机,凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型。可烫印90g/m²~350g/m²的A4各类型纸张,支持各类广泛的应用领域。 高效、稳定的进纸结构 配备44页标准ADF进纸器,支持90g/m²~350g/m²的各类纸张(普通纸、薄纸、再生纸、厚纸等),进纸通道结构稳定可靠,支持连续烫印。 * 350g/m²支持12页自动进纸 * 最大支持44页进纸容量(90g/m²)烫印面朝下 高速连续烫金 HAK180针对不同厚度、介质的纸张提供两种可选烫金速度。15ppm满足普通规格纸张的高效烫金需求,7ppm适合稍厚纸张的烫金。 10字符×2行LCD液晶屏 10字符×2行LCD液晶屏,2个自定义按键,操作直观,方便快捷。 一般参数  正常工作环境(温度): 10 ~ 32 摄氏度(50 ~ 90 华氏度) 正常工作环境(相对湿度): 20 % ~ 80 % 机器尺寸: W 384.2mm×D 330.2mm×H 356.2mm 重量(含包装箱): 16.9kg 电源: 220~240 V 消费电力(烫印中): 少于340W 消费电力(待机中): 少于7W 消费电力(关机): 少于0.04W LCD液晶屏尺寸: 48.0mm×10.9mm 节省烫金膜功能: 支持(在省膜模式中“跳过”和“中间”功能, 仅适用全幅烫金膜盒) 烫印参数  最大烫印速度 (A4): 最高达15 ppm 可选烫印速度(A4): 7 ppm 烫金机-HAK180-烫印速度调整-7PPM 烫金机-HAK180-安装耗材 烫金机-HAK180-更换耗材",
            "title": "HAK180",
            "chunks_id": None,
            "url": "https://www.brother.cn/hak/hak180",
            "source": "web",
            "score": 0.7373493866570875
        },
        {
            "content": "\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。\n\n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备的插头，然后联系 呼叫中心或您当地的 经销商。\n\n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother经销商。\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。\n\n请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅 。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。\n\n•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![⚠️ **高温警示：设备内部零件冷却前勿触碰，防止烧伤**  （图示说明：使用后设备内部部件仍达170°C/338°F，灰色标记区域为高温区；严禁在未冷却时打开前盖或触摸内部零件）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ\n\n![**图：设备内部结构示意图——手部操作部件（电源线安全警示前页）**](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)\n",
            "title": None,
            "chunks_id": 468191558938394829,
            "url": None,
            "source": "local",
            "score": 0.5320207124396898
        },
        {
            "content": ".",
            "title": "兄弟(brother) HAK180烫金机适用邀请函请柬奖状贺卡证书文印高速定制化热转印小型烫印 官方标配【图片 价格 品牌 报价】-京东",
            "chunks_id": None,
            "url": "https://item.jd.com/10082195754841.html",
            "source": "web",
            "score": 0.5082232114887758
        }
    ]

    session_id = "test_answer_session_001"
    raw_masg_list = get_recent_messages(session_id,limit=10)
    chat_history = []
    for msg in raw_masg_list:
        chat_history.append({
            "role":msg.get("role"),
            "text":msg.get("text")
        })

    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": chat_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    node_anwer_output = NodeAnswerOutput()
    result = node_anwer_output(mock_state)

    # 1. 验证 Prompt 构建
    if "prompt" in result:
        print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
        # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
    else:
        print("[FAIL] Prompt 未构建")

    # 2. 验证答案生成
    answer = result.get("answer")
    if answer and len(answer) > 10:
        print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
        print(f"答案预览: {answer}...")
    else:
        print(f"[WARN] 答案生成可能异常 (Content: {answer})")
