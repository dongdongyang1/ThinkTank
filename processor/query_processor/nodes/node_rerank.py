from processor.import_processor.base import BaseNode
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState


class NodeRerank(BaseNode):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    name :str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")

        return state