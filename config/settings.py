"""
统一配置管理入口：pydantic-settings
所有配置从环境变量/.env读取，启动时自动校验必填项
原有各config文件作为兼容层，从本settings读取
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用全局配置，alias与.env中的变量名一一对应"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # 允许同时用字段名和alias赋值
    )

    # ========== MinerU PDF解析 ==========
    mineru_api_token: str = Field(default="", alias="MINERU_API_TOKEN")
    mineru_base_url: str = Field(default="", alias="MINERU_BASE_URL")

    # ========== MinIO 对象存储 ==========
    minio_endpoint: str = Field(default="", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="", alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="", alias="MINIO_BUCKET_NAME")
    minio_img_dir: str = Field(default="kb_images", alias="MINIO_IMG_DIR")

    # ========== Milvus 向量数据库 ==========
    milvus_url: str = Field(default="", alias="MILVUS_URL")
    milvus_chunks_collection: str = Field(default="", alias="CHUNKS_COLLECTION")
    milvus_item_name_collection: str = Field(default="", alias="ITEM_NAME_COLLECTION")

    # ========== LLM 大模型 ==========
    llm_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    llm_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_vl_model: str = Field(default="", alias="VL_MODEL")
    llm_model: str = Field(default="", alias="LLM_DEFAULT_MODEL")
    llm_item_model: str = Field(default="", alias="ITEM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_DEFAULT_TEMPERATURE")
    llm_requests_per_minute: int = 15

    # ========== Embedding 向量模型 ==========
    embedding_bge_m3_path: str = Field(default="", alias="BGE_M3_PATH")
    embedding_bge_m3: str = Field(default="", alias="EMBEDDING_MODEL")
    embedding_bge_device: str = Field(default="cpu", alias="BGE_DEVICE")
    embedding_bge_fp16: bool = Field(default=False, alias="BGE_FP16")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")

    # ========== MCP 服务 ==========
    mcp_base_url: str = Field(default="", alias="MCP_DASHSCOPE_BASE_URL")
    # MCP和Reranker共用OPENAI_API_KEY，直接引用llm_api_key

    # ========== Reranker 重排序 ==========
    reranker_model: str = Field(default="", alias="TEXT_RERANK_MODEL")
    reranker_instruct: str = Field(default="", alias="TEXT_RERANK_INSTRUCT")

    # ========== Celery / Redis ==========
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")

    # ========== 文件存储 ==========
    data_based_root_dir: str = Field(default="", alias="DATA_BASED_ROOT_DIR")
    file_dir: str = Field(default="", alias="FILE_DIR")

    # ========== 导入流程常量 ==========
    mineru_max_pages: int = 200
    mineru_timeout_seconds: int = 1200

    # ========== 安全配置 ==========
    api_key: str = Field(default="",alias="API_KEY")  # API鉴权密钥，为空则不启用鉴权
    cors_origins: str = Field(default="",alias="CORS_ORIGINS") # 允许跨域的域名，逗号分隔


    # ========== token 告警阈值 ==========
    AGENT_INPUT_WARN_TOKENS: int = 8000
    KB_SEARCH_WARN_TOKENS: int = 3000

    def validate_required(self) -> list[str]:
        """校验必填配置项，返回缺失字段列表"""
        required = {
            "mineru_api_token": self.mineru_api_token,
            "mineru_base_url": self.mineru_base_url,
            "minio_endpoint": self.minio_endpoint,
            "minio_access_key": self.minio_access_key,
            "minio_secret_key": self.minio_secret_key,
            "minio_bucket_name": self.minio_bucket_name,
            "milvus_url": self.milvus_url,
            "milvus_chunks_collection": self.milvus_chunks_collection,
            "milvus_item_name_collection": self.milvus_item_name_collection,
            "llm_base_url": self.llm_base_url,
            "llm_api_key": self.llm_api_key,
        }
        return [k for k, v in required.items() if not v]


# 全局单例，启动时加载一次
settings = AppSettings()
