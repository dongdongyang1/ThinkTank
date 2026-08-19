"""Milvus配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class MilvusConfig:
    milvus_url: str
    chunks_collection: str
    item_name_collection: str


milvus_config = MilvusConfig(
    milvus_url=settings.milvus_url,
    chunks_collection=settings.milvus_chunks_collection,
    item_name_collection=settings.milvus_item_name_collection,
)
