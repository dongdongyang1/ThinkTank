from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import format_json
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search, escape_milvus_string


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """
    name : str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
         核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
         流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
         :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                       {
                           "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                           "item_names": list[str],  # step7已确认的标准化商品名列表
                       }
         :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                  {
                      "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                      # 每个元素为一条匹配的向量数据，含业务字段
                  }
        """
        logger.info(f"【{self.name}】节点逻辑")
        try:
            # 1、用户问题和已确认商品名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2. 生成向量（Dense+Sparse）
            embeddings = generate_embeddings([query])
            dense_vec = embeddings.get("dense")[0]
            sparse_vec = embeddings.get("sparse")[0]

            # 3. 获取Milvus的集合
            collection_name = milvus_config.chunks_collection

            # 4. 处理 item_names中的引号，防止注入或语法错误
            expr = None
            if item_names:
                quoted = ",".join(f'"{v}"' for v in item_names)
                expr = f"item_name in [{quoted}]"
                #expr = 'item_name like "%HAK180烫金机%"'
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 5. 构造Milvus混合搜素请求对象
            reqs = create_hybrid_search_requests(
                dense_vector = dense_vec,
                sparse_vector = sparse_vec,
                expr=expr,
                limit=10      # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            )

            # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                reqs=reqs,
                client=client,
                collection_name=collection_name,
                rank_weights = (0.8,0.2),
                output_fields=["chunks_id","content","item_name"]
            )

            # 7、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
            #hybrid_search返回格式是双层列表,外层是一个包裹列表，真正的文档数组在res[0]
            return {"embedding_chunks":res[0] if res else []}
        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {"embedding_chunks": []}



if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(format_json(result))
