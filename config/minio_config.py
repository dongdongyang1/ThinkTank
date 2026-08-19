"""MinIO配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    img_dir: str


minio_config = MinIOConfig(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    bucket_name=settings.minio_bucket_name,
    img_dir=settings.minio_img_dir,
)
