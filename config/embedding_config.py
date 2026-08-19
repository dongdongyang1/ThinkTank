"""Embedding配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool


embedding_config = EmbeddingConfig(
    bge_m3_path=settings.embedding_bge_m3_path,
    bge_m3=settings.embedding_bge_m3,
    bge_device=settings.embedding_bge_device,
    bge_fp16=settings.embedding_bge_fp16,
)
