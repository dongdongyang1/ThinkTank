from processor.import_processor.base import BaseNode
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState


class NodeWebSearchMcp(BaseNode):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    name : str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")

        return {"web_search_docs":[]}