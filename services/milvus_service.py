"""
Milvus 向量数据库服务层：封装连接、建表、幂等清理、批量插入

节点只调用本服务的公开方法，不直接接触 pymilvus SDK
"""

import logging
from typing import Callable, List, Dict, Any, Optional

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class MilvusService:
    """Milvus向量数据库通用服务"""

    def __init__(self):
        self.client = get_milvus_client()
        self.logger = logging.getLogger(__name__)
        if not self.client:
            self.logger.warning("Milvus客户端连接失败，后续操作将被跳过")

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def ensure_collection(self, collection_name: str, create_callback: Callable) -> bool:
        """
        确保集合存在，不存在则调用create_callback创建
        :param collection_name: 集合名
        :param create_callback: 无参回调，内部执行建表逻辑
        :return: True=已就绪，False=连接失败
        """
        if not self.is_connected:
            return False
        if not self.client.has_collection(collection_name):
            create_callback()
            self.logger.info(f"[Milvus] 集合 '{collection_name}' 创建完成")
        else:
            self.client.load_collection(collection_name)
            self.logger.info(f"[Milvus] 集合 '{collection_name}' 已加载")
        return True

    def delete_by_filter(self, collection_name: str, filter_expr: str) -> bool:
        """
        按过滤表达式删除数据（幂等清理）
        :param collection_name: 集合名
        :param filter_expr: 过滤表达式，如 "file_title=='xxx'"
        :return: True=成功，False=失败
        """
        if not self.is_connected:
            return False
        try:
            self.client.delete(collection_name=collection_name, filter=filter_expr)
            self.logger.info(f"[Milvus] 幂等清理完成：{collection_name} filter={filter_expr}")
            return True
        except Exception as e:
            self.logger.error(f"[Milvus] 删除失败：{e}")
            return False

    def delete_by_field_value(self, collection_name: str, field_name: str, value: str) -> bool:
        """按字段值删除（自动转义特殊字符）"""
        if not value:
            self.logger.warning(f"[Milvus] 字段值为空，跳过清理：{field_name}")
            return False
        safe_value = escape_milvus_string(value)
        return self.delete_by_filter(collection_name, f"{field_name}=='{safe_value}'")

    def insert_batch(self, collection_name: str, data: List[Dict[str, Any]]) -> Optional[dict]:
        """
        批量插入数据，返回插入结果（含insert_count和ids）
        :param collection_name: 集合名
        :param data: 待插入数据列表
        :return: 插入结果dict，失败返回None
        """
        if not self.is_connected:
            return None
        try:
            result = self.client.insert(collection_name=collection_name, data=data)
            self.client.flush(collection_name=collection_name)
            insert_count = result.get("insert_count", 0)
            self.logger.info(f"[Milvus] 批量插入完成：{insert_count}/{len(data)} 条")
            if insert_count < len(data):
                self.logger.warning(f"[Milvus] 部分插入成功：{insert_count}/{len(data)}")
            return result
        except Exception as e:
            self.logger.error(f"[Milvus] 批量插入失败（{len(data)}条）：{e}")
            return None

    def load_collection(self, collection_name: str) -> bool:
        """加载集合到内存（查询前调用）"""
        if not self.is_connected:
            return False
        try:
            self.client.load_collection(collection_name)
            return True
        except Exception as e:
            self.logger.error(f"[Milvus] 加载集合失败：{e}")
            return False

    def search(self, collection_name: str, data: List, limit: int = 5,
               output_fields: List[str] = None, filter_expr: str = "") -> List:
        """
        向量检索（供query流程使用）
        :param collection_name: 集合名
        :param data: 查询向量列表
        :param limit: 返回条数
        :param output_fields: 返回字段
        :param filter_expr: 过滤表达式
        :return: 检索结果
        """
        if not self.is_connected:
            return []
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=data,
                limit=limit,
                output_fields=output_fields or [],
                filter=filter_expr
            )
            return results
        except Exception as e:
            self.logger.error(f"[Milvus] 检索失败：{e}")
            return []
