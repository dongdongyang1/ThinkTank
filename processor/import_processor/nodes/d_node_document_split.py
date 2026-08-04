import json
import logging
import re
from pathlib import Path
from typing import Tuple,List,Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
     将长文档切分成小的 Chunks (切片) 以便检索。
    """
    name = "d_node_document_split"

    def process(self, state:ImportGraphState):
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 1. 加载并标准化输入数据
        md_content,file_title = self._step_1_inputs(state)

        # 2. 按MD标题进行初次切分
        sections,title_count,lines_count = self._step_2_split_by_title(md_content,file_title)

        # 3. 无标题场景兜底处理
        sections = self._step_3_handle_no_title(md_content,sections,title_count,file_title)

        # 4. Chunk精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)

        # 5. 输出文档切分统计信息
        self._step_5_print_stats(lines_count,sections)


        # 6 chunk结果本地JSON备份
        self._step_6_backup(state,sections)

        #写入状态字典
        state["chunks"] = sections
        return state

    def _step_1_inputs(self, state:ImportGraphState)->Tuple[str,str]:
        """
        【步骤1】获取并预处理输入数据,从 State 中提取必要的数据，并进行基础清洗。
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题
        """
        #a_node_entry节点存入的
        # 1、非空校验
        file_title = state.get("file_title")
        if not  file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="file_title", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        md_content = md_content.replace("\r\n","\n").replace("\r","\n")

        return md_content,file_title

    def _step_2_split_by_title(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = re.compile(r'^\s*#{1,6}\s+.+')
        code_pattren = re.compile(r'^(`{3,}|~{3,})')
        # 2、初始化需要的数据
        lines = content.split("\n")
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            use_title = current_title if current_title else f"【文档前置内容】{file_title}"
            sections.append({
                "title": use_title,
                # 每段时间使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 4、逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            stripped_line = line.strip()
            # 4.1 识别代码块边界 ```、~~~、````、~~~~ 等（至少 3 个连续字符）
            # 使用正则匹配：行首到行尾只有 ` 或 ~ 字符，且数量>=3
            code_block_marker_match = code_pattren.match(stripped_line)
            if code_block_marker_match:
                marker = code_block_marker_match.group(1)

                if not in_code_block:
                    # 进入代码块，记录开始的标记特征
                    in_code_block = True
                    code_block_start_marker = marker
                elif in_code_block and stripped_line.startswith(code_block_start_marker):
                    # 遇到匹配的结束标记（相同字符且相同长度）
                    in_code_block = False
                    code_block_start_marker = None

                current_lines.append(line)
                continue

            # 4.2 识别标题
            is_valid_title = (not in_code_block) and title_pattern.match(line)
            if is_valid_title:
                # 遇到标题现将上一个片段写入sections
                _flush_section()
                #初始化新的章节
                current_title = stripped_line
                current_lines = []
                title_count += 1
                self.logger.info(f"识别标题：{current_title}")
            else:
                # 普通行
                current_lines.append(line)

        _flush_section()
        self.logger.info(
            f"文档粗切（按标题切分）完成，共{len(sections)}个章节，标题数量是{title_count}，文本共有{len(lines)}行")
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self, md_content:str, sections:List[Dict[str,str]],
                                title_count:int, file_title:str)->List[Dict[str,str]]:

        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count==0:
            # 无标题情况：替换为单章节，标题为"无标题"
            self.logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
            return [{"title":"无标题","content":md_content,"file_title":file_title}]

        return sections

    def _step_4_refine_chunks(self, sections:List[Dict[str,str]])->List[Dict[str,str]]:
        """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        :param sections: 步骤3处理后的章节列表
        :return: 长度适中、低碎片化的最终Chunk列表
        """
        # 阶段1：切分超长章节 → 所有章节长度控制在最大长度内
        refined_split = []
        for sec in sections:
            # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
            refined_split.extend(self._split_long_section(sec))
            self.logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

        # 阶段2：合并过短章节 → 减少碎片化，提升后续检索/大模型调用效果
        final_sections = self._merge_short_sections(refined_split)
        self.logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

        # 阶段3：父标题兜底 → 适配Milvus向量库schema（parent_title为必填字段）
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""
        self.logger.debug(f"步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")

        return final_sections

    def _split_long_section(self, section:Dict[str,str])->List[Dict[str,str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        # 内容空值兜底：无内容直接返回原章节
        md_content = section.get("content","")
        # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
        if len(md_content)<=self.config.max_content_length:
            return [section]

        # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
        title = section.get("title","")
        # 标题前缀：带空行分隔，与正文区分开
        prefix = f"{title}\n\n" if title else ""
        # 标题前缀：带空行分隔，与正文区分开
        available_len = self.config.max_content_length - len(prefix)
        if available_len <= 0:
            self.logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
            return [section]

        # 清理正文重复标题：避免原章节中正文开头重复标题，导致子Chunk内容冗余
        body = md_content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title)+len(title):].lstrip()

        # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
        # separators：分割符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,  # 正文部分最大长度（已扣除标题）
            chunk_overlap=0,        # 无重叠：按标题切分后语义完整，无需重叠
            # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆（在 chunk_size 位置强制切断）
            # 先用第一个分隔符进行切分，切分后如果某个 Chunk 还是超过 chunk_size，则继续用下一个优先级的分隔符切分
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        # 切分正文并组装子章节（带完整元信息，便于溯源）
        sub_sections = []

        # 遍历切分后的每个文本块，idx 从 1 开始计数
        for idx,chunk in enumerate(splitter.split_text(body),start=1):
            # 清理空内容：跳过切分后的空字符串
            text = chunk.strip()
            if not text:
                continue

            # 组装子Chunk完整内容 = 标题前缀 + 切分后的正文
            full_text = (prefix+text).strip()

            # 组装子Chunk完整内容 = 标题前缀 + 切分后的正文
            sub_sections.append({
                "title": f"{title}-{idx}" if title else f"chunk-{idx}", # 父标题基础上拼接分片序号
                "content":full_text,  #子Chunk序号
                "parent_title":title,   #父章节标题（用于后续合并）
                "part": idx,   # 子Chunk序号,与标题无关，标注顺序
                "file_title":section.get("file_title"), # 所属文件标题，路径+文件名
                "end_part": idx,  # 原生单分片，起始=结束
            })

        self.logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
        return sub_sections

    def _merge_short_sections(self, sections:List[Dict[str,str]]) ->List[Dict[str,str]]:
        """
        【辅助函数】过短章节合并（减少碎片化，提升检索效果）
        核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
        :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
        :return: 合并后的Chunk列表，长度适中，保留元信息
        """
        # 边界处理：空列表直接返回，避免后续索引报错
        if not sections:
            self.logger.debug("待合并Chunk列表为空，直接返回")
            return []
        merged_sections = []   # 最终合并结果
        current_chunk = None   # 迭代累加器：保存当前待合并的Chunk

        for sec in sections:
            # 初始化：第一个Chunk直接作为当前待合并块
            if current_chunk is None:
                # current_chunk = sec
                current_chunk = {
                    "title": sec.get("title", ""),
                    "content": sec.get("content", ""),
                    "parent_title": sec.get("parent_title", ""),
                    "part": sec.get("part",1),  # 起始分片固定不变
                    "file_title": sec.get("file_title", ""),
                    "end_part": sec.get("part",1)  # 结束分片
                }
                continue

            # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
            is_current_short = len(current_chunk["content"]) < self.config.max_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")
            if is_same_parent and is_current_short:
                # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余
                parent_title = sec.get("parent_title","")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()

                # 合并内容：空行分隔，保证格式整洁
                current_chunk["content"] += "\n\n" + next_content

                # 1.更新子Chunk序号：保留最新序号，便于溯源(覆盖起始part的逻辑)
                # if "part" in sec:
                #     current_chunk["part"] = sec["part"]

                # 2、更新结束分片序号
                if "part" in sec:
                    current_chunk["end_part"] = sec["part"]
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                #current_chunk = sec
                current_chunk = {
                    "title": sec.get("title", ""),
                    "content": sec.get("content", ""),
                    "parent_title":  sec.get("parent_title", ""),
                    "part": sec.get("part",1),
                    "file_title": sec.get("file_title", ""),
                    "end_part": sec.get("part",1)
                }

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections

    def _step_5_print_stats(self, lines_count:int, sections:List[Dict[str,str]])->None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _step_6_backup(self, state:ImportGraphState, sections:List[Dict[str,str]])->None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """
        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path("D:/doc") / state.get("file_title") / "chunks.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看

            with open(backup_path,"w",encoding="utf-8") as f:
                json.dump(
                    sections,
                    f,
                    ensure_ascii=False,  # 格式化缩进，便于阅读
                    indent=2  # 格式化缩进，便于阅读
                )
                self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)



if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册_new.md"
    with open(md_path,"r",encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path":md_path,
        "md_content":md_content,
        "file_title":"hak180产品安全手册"
    }
    # 执行文档切分节点
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result,ensure_ascii=False,indent=4))


















