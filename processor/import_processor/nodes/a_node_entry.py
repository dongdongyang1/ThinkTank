from processor.import_processor.base import BaseNode
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务开发
    """
    name = "a_node_entry"  #属性
    def process(self, state: ImportGraphState) :  #方法


        return state

