import json
import logging
from typing import Any, Tuple, List, Dict

from pymilvus import DataType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from services.milvus_service import MilvusService


class NodeImportMilvus(BaseNode):
    """导入向量库节点：数据持久化，Milvus操作委托给 MilvusService"""

    name = "g_node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        if state.get("pdf_parse_error"):
            self.logger.warning(f"跳过本节点，PDF解析失败:{state['pdf_parse_error']}")
            return state

        # 1. 输入校验
        chunk_json_data, vector_dimension = self._step_1_check_input(state)

        # 2. Milvus服务+集合准备
        service = MilvusService()
        if not service.is_connected:
            raise MilvusError("Milvus 连接失败")

        collection_name = milvus_config.chunks_collection
        service.ensure_collection(
            collection_name,
            lambda: self._create_chunks_collection(collection_name, service.client, vector_dimension)
        )

        # 3. 幂等清理同文件旧数据
        self._step_3_clean_old_data(service, chunk_json_data)

        # 4. 批量插入+主键回填
        update_chunks = self._step_4_insert_data(service, chunk_json_data)

        # 5. 更新状态
        state["chunks"] = update_chunks
        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        first_chunk = chunks[0]
        if "dense_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="缺失dense_vector字段")
        if "sparse_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="缺失sparse_vector字段")

        dense_vec = first_chunk["dense_vector"]
        if not isinstance(dense_vec, list) or len(dense_vec) <= 0:
            raise StateFieldError(field_name="dense_vector", message="稠密向量为空")
        vector_dimension = len(dense_vec)
        return chunks, vector_dimension

    def _step_3_clean_old_data(self, service: MilvusService, chunk_json_data: List[Dict]):
        file_title = chunk_json_data[0].get("file_title")
        if not file_title:
            self.logger.warning("file_title 为空，跳过幂等清理")
            return
        collection_name = milvus_config.chunks_collection
        if not service.delete_by_field_value(collection_name, "file_title", file_title):
            raise MilvusError(f"Milvus 旧数据清理失败: file_title={file_title}")

    def _step_4_insert_data(self, service: MilvusService, chunk_json_data: List[Dict]) -> List[Dict]:
        # 预处理数据
        data_to_insert = []
        for item in chunk_json_data:
            item_copy = item.copy()
            item_copy.pop("chunks_id", None)
            item_copy.pop("end_part", None)
            if "part" not in item_copy:
                item_copy["part"] = 0
            if "sparse_vector" in item_copy and isinstance(item_copy["sparse_vector"], dict):
                item_copy["sparse_vector"] = {
                    int(k): v for k, v in item_copy["sparse_vector"].items()
                }
            data_to_insert.append(item_copy)

        # 批量插入
        collection_name = milvus_config.chunks_collection
        insert_result = service.insert_batch(collection_name, data_to_insert)
        if insert_result is None:
            raise MilvusError(f"Milvus 批量插入失败（{len(data_to_insert)} 条）")

        # 主键回填
        inserted_ids = insert_result.get("ids", [])
        if inserted_ids:
            if len(inserted_ids) != len(chunk_json_data):
                self.logger.warning(
                    f"回填ID数量不匹配: 返回{len(inserted_ids)}个，预期{len(chunk_json_data)}个"
                )
            for idx, item in enumerate(chunk_json_data):
                item["chunks_id"] = inserted_ids[idx]
        return chunk_json_data

    def _create_chunks_collection(self, collection_name, milvus_client, vector_dimension):
        """建表回调：仅在集合不存在时调用"""
        schema = milvus_client.create_schema(auto_id=True, enabled_dynamic_field=True)
        schema.add_field(field_name="chunks_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_name="dense_vector_index",
            index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector", index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        milvus_client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )
        milvus_client.load_collection(collection_name)
        self.logger.info(f"集合 '{collection_name}' 创建并加载完成")


if __name__ == "__main__":
    setup_logging()
    # 调试入口
    init_state = {"chunks": []}
    node = NodeImportMilvus()
    result = node(init_state)
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
