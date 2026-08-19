"""Reranker配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class RerankerConfig:
    text_rerank_api_key: str  # DashScope API Key
    text_rerank_model: str  # 模型名称
    text_rerank_instruct: str  # 是否使用指令


reranker_config = RerankerConfig(
    text_rerank_api_key=settings.llm_api_key,  # Reranker与LLM共用OPENAI_API_KEY
    text_rerank_model=settings.reranker_model,
    text_rerank_instruct=settings.reranker_instruct,
)
