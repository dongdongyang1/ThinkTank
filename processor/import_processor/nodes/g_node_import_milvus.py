import json
import logging
from typing import Any, Tuple, List,Dict

from pymilvus import DataType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "g_node_import_milvus"

    def process(self, state: ImportGraphState)->ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：删除同file_title旧数据，避免重复存储
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        异常处理：
            任一步骤失败抛出异常，终止节点执行，保证数据不脏写

        必要参数：chunks
        新参数：chunks字段回填chunk_id

            :param state: 工作流状态对象
            :return: 更新后的状态对象
        """

        # 1. 输入数据有效性校验
        chunk_json_data ,vector_dimension = self._step_1_check_input(state)

        # 2. milvus客户端连接+集合准备（自动建表）
        client = self._step_2_prepare_collection(vector_dimension)

        # 3. 幂等性处理 - 清理同file_title旧数据
        self._step_3_clean_old_data(client,chunk_json_data)

        # 4. 批量插入数据+主键chunlk_id回填
        update_chunks = self._step_4_insert_data(client,chunk_json_data)

        # 5. 更新全局状态，将回填后的切片回传下游
        state["chunks"] = update_chunks


        return state

    def _step_1_check_input(self, state:Dict[str,Any])->Tuple[List[Dict[str,Any]],int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            tuple - (校验通过的切片列表, 稠密向量维度)
        异常：
        任一校验项不通过，抛出StateFieldError终止入库流程，避免脏数据处理

        """

        chunks = state.get("chunks")

        # 检验1 ：chunks非空
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks,list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 检验2 ：切片包含dense_vector字段
        first_chunk = chunks[0]
        if "dense_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 检验3 ：切片包含sparse_vector字段
        if "sparse_vector" not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 4. 提取向量维度
        dense_vec = first_chunk["dense_vector"]
        if not isinstance(dense_vec, list) or len(dense_vec) <= 0:
            raise StateFieldError(field_name="dense_vector", message="稠密向量为空，无法创建Milvus集合")
        vector_dimension = len(first_chunk["dense_vector"])
        return chunks,vector_dimension

    def _step_2_prepare_collection(self, vector_dimension:int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端，验证连接有效性
            2. 集合不存在则自动创建（Schema+索引），存在则直接复用
        参数：
            vector_dimension: int - 稠密向量维度（步骤1提取）
        返回：
            MilvusClient - 已连接、集合准备完成的客户端实例
        异常：
            户端获取失败/集合名称未配置，抛出异常终止流程
         """
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 2. 集合不存在则创建
        collection_name = milvus_config.chunks_collection

        # if not milvus_client.has_collection(collection_name):
        #     self._create_chunks_collection(collection_name,milvus_client,vector_dimension)
        # else:
        #     # 集合已存在但可能未加载（如 Milvus 重启后），显式加载
        #     milvus_client.load_collection(collection_name)
        #     self.logger.info(f"集合 '{collection_name}' 已加载")


        return milvus_client

    def _create_chunks_collection(self, collection_name, milvus_client, vector_dimension):
        # 1. 创建schema
        schema = milvus_client.create_schema(auto_id = True,enabled_dynamic_field = True)
        # 2. 创建列
        schema.add_field(field_name="chunks_id",datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name="content",datatype=DataType.VARCHAR,max_length=65535) # 切片内容
        schema.add_field(field_name="title",datatype=DataType.VARCHAR,max_length=512) # 切片标题
        schema.add_field(field_name="parent_title",datatype=DataType.VARCHAR,max_length=512) # 父标题
        schema.add_field(field_name="part",datatype=DataType.INT8) # 分片编号，顺序
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512) # 源文件标题(无后缀)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=512) # 商品名称
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=vector_dimension) #稠密向量
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR) #稀疏向量
        # 3.创建索引
        index_params = milvus_client.prepare_index_params()

        # 稠密向量索引：AUTOINDEX自动选最优索引类型+余弦相似度（语义检索常用
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+内积（IP），适配稀疏向量检索
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo":"DAAT_MAXSCORE","normalize":True,"quantization":"none"}

        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        milvus_client.load_collection(collection_name)
        self.logger.info(f"集合 '{collection_name}' 创建并加载完成")

    def _step_3_clean_old_data(self, client, chunk_json_data):
        # 1. 获取查询条件
        file_title = chunk_json_data[0].get("file_title")

        # 2. 执行幂等清理
        if not file_title:
            self.logger.warning("file_title 为空，跳过幂等清理")
            return
        self._clear_chunks_by_file_title(client,file_title)

    def _clear_chunks_by_file_title(self, client, file_title):
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(collection_name=milvus_config.chunks_collection,filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    def _step_4_insert_data(self, client, chunk_json_data):
        """
        步骤4：批量插入切片数据到Milvus+主键回填
        核心逻辑：
            1. 批量插入数据：提升入库效率，减少Milvus连接次数
            2. 回填chunk_id：将Milvus生成的自增主键回填到切片，供下游业务使用
        参数：
            client - MilvusClient实例
            chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
        返回：
            List[Dict[str, Any]] - 回填了chunk_id的切片列表
        """

        # 1. 预处理数据：移除手动chunk_id，避免与Milvus自增主键冲突
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

        # 2. 执行批量插入
        try:
            insert_result = client.insert(
                collection_name=milvus_config.chunks_collection,
                data=data_to_insert
            )
        except Exception as e:
            raise MilvusError(
                message=f"Milvus 批量插入失败（{len(data_to_insert)} 条数据）: {e}",
                node_name=self.name,
                cause=e,
            )

        # pymilvus新版返回字典，用key取值
        insert_count = insert_result["insert_count"]
        self.logger.info(f"Milvus 批量插入完成: {insert_count}/{len(data_to_insert)} 条")

        if insert_count == 0:
            self.logger.warning(f"Milvus 插入数量为 0，请检查数据与集合状态")
        elif insert_count < len(data_to_insert):
            self.logger.warning(f"Milvus 部分插入成功: {insert_count}/{len(data_to_insert)}")

        # 主键从ids字段获取
        inserted_ids = insert_result["ids"]
        if inserted_ids:
            if len(inserted_ids) != len(chunk_json_data):
                self.logger.warning(
                    f"回填 ID 数量不匹配: 返回 {len(inserted_ids)} 个 ID，预期 {len(chunk_json_data)} 个"
                )
            for idx, item in enumerate(chunk_json_data):
                item["chunks_id"] = inserted_ids[idx]
        return chunk_json_data



if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\doc\hak180产品安全手册\state_vector.json"
    with open(json_path,"r",encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks":state.get("chunks")
    }

    node_imort_milvus = NodeImportMilvus()
    result = node_imort_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))