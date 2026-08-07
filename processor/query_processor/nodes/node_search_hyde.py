from processor.import_processor.base import BaseNode
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState


class NodeSearchHyde(BaseNode):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    name : str = "node_search_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")

        return {"hyde_embedding_chunks" : []}