from processor.import_processor.base import BaseNode
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState


class NodeAnswerOutput(BaseNode):
    """
    节点功能: 答案生成
    """

    name :str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")

        return state