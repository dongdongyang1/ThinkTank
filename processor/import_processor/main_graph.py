import json
import sys
from pathlib import Path

# 获取当前文件绝对路径，向上3层锁定项目根目录
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from langgraph.constants import END
from langgraph.graph import StateGraph
from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.a_node_entry import NodeEntry
from processor.import_processor.nodes.b_node_pdf_to_md import NodePDFToMD
from processor.import_processor.nodes.c_node_md_img import NodeMDImg
from processor.import_processor.nodes.d_node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.e_node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.f_node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.g_node_import_milvus import NodeImportMilvus
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    """
    知识库导入工作流
    """

    def __init__(self,config=None):
        self.__compiled_graph = None


    @property
    def graph(self):
        """
        懒加载：只在第一次使用时编译图
        """
        if self.__compiled_graph is None:
            self.__compiled_graph = self.build_graph()
        return self.__compiled_graph


    @staticmethod
    def route_after_entry(state:ImportGraphState)->str:
        """
        入口节点后的条件路由函数
        :param state: 当前状态
        :return: 下一个节点名称
        """
        if state.get("is_pdf_read_enabled"):
            return "b_node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "c_node_md_img"
        else:
            return END

    @staticmethod
    def route_after_pdf_parse(state:ImportGraphState)->str:
        """
        PDF解析节点后的条件路由函数
        解析失败则直接结束，避免后续节点因md_path为空连锁报错
        :param state: 当前状态
        :return: 下一个节点名称
        """
        if state.get("pdf_parse_error"):
            return END
        return "c_node_md_img"


    def build_graph(self):
        """
        创建图结构
        :return:编译后的图
        """

        # 1. 初始化LangGraph状态图
        graph = StateGraph(ImportGraphState)

        # 2. 注册节点到工作流
        graph.add_node("a_node_entry",NodeEntry())
        graph.add_node("b_node_pdf_to_md",NodePDFToMD())
        graph.add_node("c_node_md_img",NodeMDImg())
        graph.add_node("d_node_document_split",NodeDocumentSplit())
        graph.add_node("e_node_item_name_recognition",NodeItemNameRecognition())
        graph.add_node("f_node_bge_embedding",NodeBGEEmbedding())
        graph.add_node("g_node_import_milvus",NodeImportMilvus())

        # 3. 设置入口节点
        graph.set_entry_point("a_node_entry")

        # 4. 注册条件边
        graph.add_conditional_edges(
            "a_node_entry",
            self.route_after_entry,
            {
                "b_node_pdf_to_md": "b_node_pdf_to_md",
                "c_node_md_img":"c_node_md_img",
                END:END
            }
        )


        # 5. 注册工作流
        # PDF解析后条件路由：失败直接结束，成功才进图片处理
        graph.add_conditional_edges(
            "b_node_pdf_to_md",
            self.route_after_pdf_parse,
            {
                "c_node_md_img": "c_node_md_img",
                END: END
            }
        )
        graph.add_edge("c_node_md_img","d_node_document_split")
        graph.add_edge("d_node_document_split","e_node_item_name_recognition")
        graph.add_edge("e_node_item_name_recognition","f_node_bge_embedding")
        graph.add_edge("f_node_bge_embedding","g_node_import_milvus")
        graph.add_edge("g_node_import_milvus",END)

        # 6. 编译工作流
        graph_compiled = graph.compile()
        return  graph_compiled


    def run(self,state:ImportGraphState,stream:bool = False):
        """
        统一执行入口，支持切换invoke/stream
        :param state: 初始状态
        :param stream: 是否流式输出
        :return: 执行结果
        """

        if stream:
            #return self.graph.stream(state,stream_mode="values")
            return self.graph.stream(state,stream_mode="updates")
        else:
            return self.graph.invoke(state)


if __name__=="__main__":

    # 启用日志
    setup_logging()

    # 定义初始状态
    init_state= {"import_file_path":r"D:\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"}
    workflow = KBImportWorkflow()

    # 方式1：实例化后使用（流式输出）
    for event in workflow.run(init_state,stream=True):
        print(f"state:{event}")

    # 方式2：非流式输出
    # final_state = workflow.run(init_state,stream=False)
    # json.dumps()把 Python对象（字典/列表）转换成JSON格式字符串
    # print(json.dumps(final_state,ensure_ascii=False,indent=4))

    #打印编译后的图结构
    workflow.graph.get_graph().print_ascii()



