"""MCP配置（兼容层：从统一settings读取）"""
from dataclasses import dataclass
from config.settings import settings


@dataclass
class McpConfig:
    mcp_base_url: str
    api_key: str


mcp_config = McpConfig(
    mcp_base_url=settings.mcp_base_url,
    api_key=settings.llm_api_key,  # MCP与LLM共用OPENAI_API_KEY
)
