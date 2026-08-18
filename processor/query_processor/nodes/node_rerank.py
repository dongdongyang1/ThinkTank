from typing import List, Dict, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState
from utils.json_format_utils import format_json
from utils.reranker_http_utils import rerank_documents

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK : int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK : int = 3

# 断崖阈值（绝对，判断高分文档）
RERANK_GAP_ABS : float = 0.5
# 断崖阈值（相对，判断低分文档）
RERANK_GAP_RATIO : float = 0.25

class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    name :str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
            执行重排序
            流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
            :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
            :return: 更新后的 state，包含 reranked_docs
        """
        logger.info(f"【{self.name}】节点逻辑")
        # 1. 合并多源文档
        merge_muti_docs : List[Dict[str,Any]] = self._step_1_merge_multi_souce_docs(state)

        # 2. Rerank精排（精排打分）
        reranked_docs : List[Dict[str,Any]] = self._step_2_rerank_merged_docs(state,merge_muti_docs)

        # 3. 动态 Top_k截取（断崖检测）
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state["reranked_docs"] = cutoff_docs

        # 5. 返回state
        return state

    def _step_1_merge_multi_souce_docs(self, state:QueryGraphState)->List[Dict[str,Any]]:
        """合并本地 RRF 结果和网络搜索结果为统一格式"""
        final_docs = []
        # 1. 获取本地 RRF 的文档
        for rrf_doc in state.get("rrf_chunks",[]):
            content = rrf_doc.get("content", "").strip()
            if not content:  # 内容为空直接跳过
                continue
            format_rrf_doc = {
                "content" : rrf_doc.get("content"),
                "title" : rrf_doc.get("title"),
                "chunks_id" : rrf_doc.get("chunks_id"),
                "url" : None,
                "source" : "local"
            }
            final_docs.append(format_rrf_doc)
        # 2. 获取 web 远程的文档
        for web_doc in state.get("web_search_docs",[]):
            content = web_doc.get("snippet", "").strip()
            if not content:  # 网页摘要空，丢弃这条文档
                continue
            format_web_doc = {
                "content" : web_doc.get("snippet"),
                "title" : web_doc.get("title"),
                "chunks_id" : None,
                "url" : web_doc.get("url"),
                "source" : "web"
            }
            final_docs.append(format_web_doc)
        logger.info(f"合并后有效文档数量：{len(final_docs)}")
        return final_docs

    def _step_2_rerank_merged_docs(self, state:QueryGraphState, merge_muti_docs:List[Dict[str,Any]])->List[Dict[str,Any]]:
        """使用 Reranker 模型对文档进行精排"""
        if not merge_muti_docs:
            logger.warning("合并文档为空，跳过重排序调用")
            return []
        try:
            user_query = state.get("rewritten_query")
            # 获取文档列表的conten字段组成列表
            contents = [doc.get("content") for doc in merge_muti_docs]
            # 调用Rerank模型：交叉编码器（精排阶段）
            # Query 和 Document 联合编码，精度更高
            rerank_score = rerank_documents(user_query,contents)
            #**doc：把doc字典里面所有 key‑value 全部摊开
            scotrs_docs = [{**doc,"score":score} for doc,score in zip(merge_muti_docs,rerank_score)]
            sorted_score_docs = sorted(
                scotrs_docs,
                key=lambda x:x["score"],
                reverse=True
            )
            return sorted_score_docs
        except Exception as e:
            logger.error(f"重排序失败：{str(e)}")
            return [{**doc,"score":None} for doc in merge_muti_docs]

    def _step_3_cliff_cutoff(self, reranked_docs:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """断崖检测截断：相邻得分差距超过阈值时截断。"""
        if not reranked_docs:
            return []
        upper_bound = min(RERANK_MAX_TOPK,len(reranked_docs))
        lower_bound = min(RERANK_MIN_TOPK,len(reranked_docs))

        # 默认值：取满硬上限（最多10条）
        cutoff_pos = upper_bound
        # 遍历范围：从min_topk-1到max_topk-2（索引从0开始），检测相邻两个文档的分数差
        # 例：min_topk=3，max_topk=10 → 遍历i=2,3,4,5,6,7,8（对应第3~9条文档，检测与下一条的差距
        for idx in range(lower_bound-1,upper_bound-1):
            current_score = reranked_docs[idx].get("score")
            next_score = reranked_docs[idx+1].get("score")

            if current_score is None or next_score is None:
                continue

            # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            abs_gap = current_score-next_score
            # 计算相对差距：绝对差距 / 当前文档分数（+1e-6避免除数为0/极小值，防止程序报错）
            # 1e-6 是 Python 中科学计数法的写法，等价于 0.000001（10 的负 6 次方，也就是百万分之一）。
            rel_gap = abs_gap/(abs(current_score)+1e-6)
            # 触发断崖截断条件：绝对差距≥绝对阈值 OR 相对差距≥相对阈值
            # 满足任一条件，说明下一条文档相关性骤降，截断在当前位置
            if abs_gap >=RERANK_GAP_ABS or rel_gap>=RERANK_GAP_RATIO:
                cutoff_pos = idx + 1
                logger.debug(f"断崖检测: 位置 {idx + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break
        return reranked_docs[:cutoff_pos]  #算上了idx


if __name__ == "__main__" :
    mock_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "rrf_chunks": [
            {
                "chunks_id": 468191558938394830,
                "content": "\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。\n\n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。\n\n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。\n\n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。\n\n•\t请勿将任何物体压在电源线上。\n\n•\t请勿将本设备放在人们可能踏过电源线的位置。\n\n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，否则电源线可能会磨损或损坏。\n\n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏 磨损的部分。\n\n•\t请勿让设备压在电源线上。\n\n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。\n\n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照 正确安装。\n\n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。\n\n•\tBrother 强烈建议您不要使用任何类型的延长线。\n\n•\t定期拔出电源插头进行清洁。使用干布清洁插头插脚根部以及插脚之间的位置。如果电源插头长时间插入在电源插座中，灰尘会堆积在插头插脚周围，这可能会导致短路，从而引起火灾。\n\n•\t本设备装有接地的插头。此插头只能插入接地的电源插座中。这是一项安全功能。如果您无法将插头插入到插座中，请让电工更换过时的插座。请勿试图破坏接地插头的作用。\n\n不遵守说明和警告可能导致人员中度或严重受伤。遵守这些指引以避免人员受伤。\n",
                "item_name": "HAK180烫金机"
            },
            {
                "item_name": "HAK180烫金机",
                "chunks_id": 468191558938394829,
                "content": "\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。\n\n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。\n\n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n•\t请在以下环境使用本设备：温度保持在 10 °C 和 32 °C 之间，湿度保持在 20% 和 80% 之间，无冷凝。\n\n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。\n\n•\t请勿将设备放在加热器、空调、电风扇或水附近。\n\n否则当水（包括加热 空调 通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。\n\n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的 Brother 经销商。\n\n如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备的插头，然后联系 呼叫中心或您当地的 经销商。\n\n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother经销商。\n\n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。\n\n请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅 。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。\n\n•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。\n\n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![⚠️ **高温警示：设备内部零件冷却前勿触碰，防止烧伤**  （图示说明：使用后设备内部部件仍达170°C/338°F，灰色标记区域为高温区；严禁在未冷却时打开前盖或触摸内部零件）](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/f3349cded08d6686a93d0a81b9a64ec1e50d9a82cbb88541b37027f085813a15.jpg)  \n儎⑟ഴḽ䆜઀ᛞ࠽व䀜᪮儎⑟Ⲻ䇴༽䜞ԬȾ\n\n![**图：设备内部结构示意图——手部操作部件（电源线安全警示前页）**](http://192.168.204.130:9002/think-tank/upload-images/hak180产品安全手册/501bb8d2d681e4502d87badb15a68939eadfa086d309c3599f1c36b0bc559177.jpg)\n"
            },
        ],
        "web_search_docs": [
            {
                "title": "HAK180",
                "url": "https://www.brother.cn/hak/hak180",
                "snippet": "烫金机 可选7PPM烫金速度  无版烫印  配备最大44页标准ADF进纸器  支持省膜模式  10字符x2行LCD液晶屏  HAK180烫金机,凭借其高速、高品质、以及出色的细节小字烫印效果,成为定制化专属机型。可烫印90g/m²~350g/m²的A4各类型纸张,支持各类广泛的应用领域。 高效、稳定的进纸结构 配备44页标准ADF进纸器,支持90g/m²~350g/m²的各类纸张(普通纸、薄纸、再生纸、厚纸等),进纸通道结构稳定可靠,支持连续烫印。 * 350g/m²支持12页自动进纸 * 最大支持44页进纸容量(90g/m²)烫印面朝下 高速连续烫金 HAK180针对不同厚度、介质的纸张提供两种可选烫金速度。15ppm满足普通规格纸张的高效烫金需求,7ppm适合稍厚纸张的烫金。 10字符×2行LCD液晶屏 10字符×2行LCD液晶屏,2个自定义按键,操作直观,方便快捷。 一般参数  正常工作环境(温度): 10 ~ 32 摄氏度(50 ~ 90 华氏度) 正常工作环境(相对湿度): 20 % ~ 80 % 机器尺寸: W 384.2mm×D 330.2mm×H 356.2mm 重量(含包装箱): 16.9kg 电源: 220~240 V 消费电力(烫印中): 少于340W 消费电力(待机中): 少于7W 消费电力(关机): 少于0.04W LCD液晶屏尺寸: 48.0mm×10.9mm 节省烫金膜功能: 支持(在省膜模式中“跳过”和“中间”功能, 仅适用全幅烫金膜盒) 烫印参数  最大烫印速度 (A4): 最高达15 ppm 可选烫印速度(A4): 7 ppm 烫金机-HAK180-烫印速度调整-7PPM 烫金机-HAK180-安装耗材 烫金机-HAK180-更换耗材"
            },
            {
                "title": "兄弟(brother) HAK180烫金机适用邀请函请柬奖状贺卡证书文印高速定制化热转印小型烫印 官方标配【图片 价格 品牌 报价】-京东",
                "url": "https://item.jd.com/10082195754841.html",
                "snippet": "."
            },
        ],
    }
    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    logger.info(format_json(result))