# ThinkTank

> 基于 RAG + Agent 的企业智能知识库系统，支持 PDF 文档智能解析入库与多轮对话检索。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-3.0-green)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能特性

- **文档智能导入**：PDF → MinerU 解析 → 图片提取 → 语义切分 → 产品名识别 → BGE-M3 向量化 → Milvus 入库，全流程异步化
- **混合检索问答**：HyDE 查询扩展 → 向量检索 → RRF 融合排序 → Rerank 重排 → 产品名确认 → LLM 答案生成
- **ReAct Agent**：集成知识库检索（KB Search）与联网搜索（Web Search / MCP）工具，支持多步推理
- **流式响应**：基于 SSE（Server-Sent Events）的逐字输出
- **Web 管理界面**：内置文档导入页与对话聊天页
- **可扩展架构**：配置化管理 LLM、Embedding、Reranker、向量库等组件，便于切换

## 技术栈

| 类别 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent / 工作流 | LangGraph + LangChain |
| 向量数据库 | Milvus (BGE-M3) |
| 对象存储 | MinIO |
| 对话历史 | MongoDB |
| 异步任务 | Celery + Redis |
| 文档解析 | MinerU (PDF → Markdown) |
| LLM | DashScope / OpenAI 兼容接口 |
| 数据校验 | Pydantic + pydantic-settings |

## 项目结构

```
ThinkTank/
├── config/                  # 全局配置（LLM、Milvus、MinIO、Celery 等）
├── processor/               # 核心业务流水线
│   ├── import_processor/    # 文档导入流水线（7 个节点）
│   ├── query_processor/     # 检索问答流水线（8 个节点）
│   └── agent_processor/     # ReAct Agent（KB + Web 工具）
├── services/                # 外部服务封装（Milvus、MinIO、MinerU 等）
├── utils/                   # 工具函数（Embedding、LLM、SSE、Celery 任务等）
├── web/                     # Web 层
│   ├── api/                 # FastAPI 接口（导入 / 查询）
│   └── page/                # 前端页面（chat.html / import.html）
├── tool/                    # 辅助脚本（模型下载、PDF 切分等）
├── tests/                   # 单元测试
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose（用于启动 Milvus、MinIO、MongoDB、Redis）
- 至少 4GB 显存（用于 BGE-M3 嵌入模型，或使用 API 模式）

### 1. 克隆项目

```bash
git clone https://github.com/your-username/ThinkTank.git
cd ThinkTank
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制模板并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下项：

```env
# LLM
DASHSCOPE_API_KEY=your_api_key
LLM_MODEL=qwen-plus

# Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# MinIO
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key

# MongoDB
MONGO_URI=mongodb://127.0.0.1:27017

# Redis (Celery Broker)
REDIS_URL=redis://127.0.0.1:6379/0
```

### 4. 启动依赖服务

使用 Docker Compose 启动中间件：

```bash
docker compose up -d
```

### 5. 启动 Celery Worker（异步任务）

```bash
celery -A utils.celery_tasks worker --loglevel=info
```

### 6. 启动 Web 服务

```bash
uvicorn web.api:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 即可使用。

## API 接口

### 文档导入

```http
POST /api/import
Content-Type: multipart/form-data

file: @document.pdf
```

返回任务 ID，可通过任务 ID 查询导入进度。

### 对话查询

```http
POST /api/query
Content-Type: application/json

{
  "question": "这个产品的保修期是多久？",
  "session_id": "optional-session-id"
}
```

支持 SSE 流式响应。

## 核心流程

### 导入流水线

```
PDF 上传 → MinerU 解析(PDF→MD) → 图片提取上传 MinIO
→ 文档语义切分 → LLM 产品名识别 → BGE-M3 向量化 → 写入 Milvus
```

### 检索流水线

```
用户问题 → HyDE 生成假设文档 → 向量检索
→ RRF 融合排序 → Rerank 重排 → 产品名确认
→ LLM 生成答案 → (可选) Agent 联网搜索 → SSE 流式输出
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent 编排框架
- [Milvus](https://github.com/milvus-io/milvus) - 向量数据库
- [MinerU](https://github.com/opendatalab/MinerU) - PDF 文档解析
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) - 多语言嵌入模型
