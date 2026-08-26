import logging
from asyncio.log import logger
from utils.embedding_utils import get_bge_m3_ef


# 初始化日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 加载BGE-M3模型单例
model = get_bge_m3_ef()

# 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
texts = ["测试","hello"]
embeddings = model.encode_documents(texts)

logger.debug(f"获取成功！")
logger.debug(embeddings)

# 稠密向量获取
dense_obj = embeddings["dense"]
dense_list = [emb.tolist() for emb in dense_obj]
logger.debug(dense_list)

# 稀疏向量获取：解析为字典格式（适配序列化/存储）
sparse_obj = embeddings["sparse"]
processed_sparse = []
for i in range(len(texts)):

    # 提取第i个文本的稀疏向量索引
    sparse_indices = sparse_obj.indices[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    # 提取第i个文本的稀疏向量权重
    sparse_data = sparse_obj.data[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    # 构造{特征索引: 归一化权重}的稀疏向量字典
    # 把两个列表“打包”成一个字典（Milvus 数据库的要求）
    sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
    processed_sparse.append(sparse_dict)