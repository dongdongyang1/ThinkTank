from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF转MarkDown节点：PDF结构化解析
    """
    name = "b_node_pdf_to_md"

    def process(self, state:ImportGraphState):


        return state