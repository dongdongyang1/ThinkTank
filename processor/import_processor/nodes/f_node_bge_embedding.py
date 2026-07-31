from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用BGE-M3模型将文本转换为向量
    """

    name = "f_node_bge_embedding"

    def process(self, state: ImportGraphState):

        return state
