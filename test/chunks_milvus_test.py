from utils.milvus_utils import get_milvus_client
from pymilvus import DataType
from config.milvus_config import milvus_config

cli = get_milvus_client()
col = milvus_config.chunks_collection
if cli.has_collection(col):
    cli.drop_collection(col)

schema = cli.create_schema(auto_id=True, enabled_dynamic_field=True)
schema.add_field("chunks_id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("content", DataType.VARCHAR, max_length=65535)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

idx = cli.prepare_index_params()
idx.add_index("dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
idx.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
cli.create_collection(col, schema=schema, index_params=idx)
print("建表成功")