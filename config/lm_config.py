"""LLM配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    vl_model: str
    llm_model: str
    item_model: str
    llm_temperature: float
    requests_per_minute: int


lm_config = LLMConfig(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    vl_model=settings.llm_vl_model,
    llm_model=settings.llm_model,
    item_model=settings.llm_item_model,
    llm_temperature=settings.llm_temperature,
    requests_per_minute=settings.llm_requests_per_minute,
)
