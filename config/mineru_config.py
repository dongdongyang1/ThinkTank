"""MinerU配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class MineruConfig:
    api_token: str
    base_url: str


mineru_config = MineruConfig(
    api_token=settings.mineru_api_token,
    base_url=settings.mineru_base_url,
)
