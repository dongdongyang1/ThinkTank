# ThinkTank

> **DAG → ReAct Agent 架构升级** —— 基于 RAG + ReAct Agent 的企业智能知识库系统。PDF 一键智能入库，多轮对话精准检索，LLM 自主规划推理、动态调用工具，长短时记忆跨会话沉淀，多层规则遏制模型幻觉。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct%20Agent-orange)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-3.0-green)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 核心亮点

### 一、ReAct Agent：从固定 DAG 到自主推理

传统 RAG 用 DAG 流水线串联检索步骤，查询路径写死、灵活度差。ThinkTank 升级为 **ReAct（Reasoning + Acting）Agent 架构**：

- **自主规划**：LLM 每轮自主推理"下一步该做什么"——直接回答 / 调用知识库检索 / 调用联网搜索，而非按预设流程硬走
- **动态工具调用**：封装 KB Search（知识库混合检索）与 Web Search（MCP 联网搜索）两大工具，LLM 按需选择
- **多步推理链**：最多 5 轮 ReAct 循环（agent → tools → agent），复杂问题拆解为多步检索与推理，检索不到就换工具/换关键词再试，告别"硬答"
- **循环上限保护**：达到最大轮数时强制生成最终答案，防止无限循环

### 二、长短时记忆：对话上下文 + 跨会话用户画像

#### 短期记忆（工作记忆）
- 基于 LangGraph `AgentState`，完整保存对话历史、Agent 思考过程、工具调用结果
- 历史对话 token 预算控制（4000 tokens），user 消息全保留，assistant 长回答自动压缩为一句话摘要，超预算从最旧开始丢弃
- 临时 session_id 隔离，每次工具调用使用独立 session，避免历史串扰

#### 长期记忆（跨会话用户画像）
- **MongoDB 存储，按 user_id 隔离**（user_id 前端生成，跨会话不变，不依赖真实身份）
- **四类记忆**：preference（偏好，细分 style 风格 / content 内容要求）、fact（重要事实）、device（设备信息）、history（重要历史）
- **LLM 自动提取**：每轮对话结束后用 qwen-flash 自动提取值得记住的信息，短句压缩 + 类型/重要性标注，一步完成
- **偏好矛盾检测**：新偏好入库时用 LLM 判断与旧偏好是否真正矛盾，只降级矛盾的旧偏好，避免误伤
- **内容哈希去重**：同 user + 同内容哈希 → 刷新时间并提升重要性，不重复堆叠
- **重要性评分**（1-10）：加载时按重要性降序、时间降序排列，低于阈值不注入
- **时间衰减 + 自动淘汰**：超过 180 天的记忆自动失效；单用户超过 50 条时删除最旧、最低重要性的超额部分，防止无限膨胀
- **token 预算保护**：记忆区累计超过 800 tokens 即停止加载，防止 prompt 膨胀

### 三、幻觉防控：多层规则 + 检索增强 + 来源追溯

针对企业产品文档中型号高度相似（如 W585 与 W585X）导致的张冠李戴问题，设计多层幻觉防控机制：

#### 系统提示词硬约束
- **最高优先级规则**：产品相关问题必须先调用 kb_search 检索，禁止直接回答（即使 LLM "觉得自己知道"）
- **型号归属规则**：必须用与用户问题中指定型号**完全匹配**的文档回答，禁止用其他型号的内容张冠李戴
- **否定查询规则**：问题型号文档中未提及的功能 → 如实回答"未提及"，**严禁**因为其他型号有该功能就回答"支持"
- **对比查询规则**：对比问题必须分开陈述（先 A 后 B），每个型号的事实只能来自该型号自己的文档，禁止混着引用
- **工具调用禁令**：禁止用文字模拟工具调用（如"正在检索..."），必须通过 function calling 机制完成

#### 检索层防控
- **产品名确认节点**：LLM 提取 + 向量检索对齐（阈值 0.85 高置信确认 / 0.6 候选反问）+ 规则兜底，三重保障型号识别准确
- **单型号问题隔离**：防止宽泛关键词（如"擎云"）把同系列多个易混淆型号都带进来，只保留 query 中明确出现的型号
- **query 型号补全**：LLM 漏掉的型号从商品名全表补回，保证召回率

#### 输出层防控
- **来源标注**：回答时必须标注原 PDF 文档名称（如"根据《W585用户指南》"）
- **联网内容标注**：基于联网搜索的内容末尾标注"（以上信息来自互联网搜索，仅供参考）"，与知识库内容明确区分
- **纯文本输出约束**：禁止 Markdown 语法，避免格式混乱导致信息误读

### 四、混合检索引擎

HyDE 查询扩展 → 稠密+稀疏混合检索（BGE-M3）→ RRF 融合排序 → Rerank 重排 → 产品名确认，层层过滤精准命中。

- **双向量检索**：BGE-M3 同时生成稠密向量（语义）+ 稀疏向量（关键词），稠/稀疏权重 0.8/0.2
- **HyDE 查询扩展**：LLM 先生成假设答案，再用假设答案向量化检索，提升语义匹配度
- **RRF 融合排序**：稠密+稀疏两路结果用 Reciprocal Rank Fusion 融合，兼顾语义与关键词
- **Rerank 重排**：用交叉编码器对融合结果精排，提升 Top-K 准确率
- **Milvus 双集合设计**：产品名识别集合用 IVF_FLAT（nlist=128，精确可控），文档块集合用 AUTOINDEX（自动调优），均搭配 SPARSE_INVERTED_INDEX 稀疏倒排索引

### 五、PDF 智能入库

MinerU 解析 → 图片提取上传 MinIO → 多模态图片理解（VL 模型生成摘要）→ 语义切分 → LLM 产品名识别 → BGE-M3 向量化 → Milvus 入库，全流程 Celery 异步化。

- **超大 PDF 切片处理**：超过 200 页的 PDF 自动切片，避免 MinerU 超时
- **图片理解**：VL 模型为每张图片生成中文摘要，替换原始图片引用，让纯文本检索也能命中图片内容
- **图片 URL 兼容**：MinIO 路径含中文/括号/特殊符号时自动 URL 编码，前端渲染零异常

### 六、安全与隐私

- **API 鉴权**：可配置 API_KEY 对接口进行鉴权，为空则不启用（开发模式）
- **CORS 跨域配置**：可配置允许跨域的域名列表，生产环境限制来源
- **用户数据隔离**：user_id 隔离长期记忆，session_id 隔离对话历史，不同用户数据互不干扰
- **配置化密钥管理**：所有 API Key、数据库密码从环境变量读取，不硬编码在代码中
- **无真实身份依赖**：user_id 由前端生成，不收集用户真实身份信息

### 更多特性

- **SSE 流式响应**：答案逐字输出，对话体验流畅
- **Web 管理界面**：内置文档导入页 + 对话聊天页，开箱即用
- **配置化架构**：LLM、Embedding、Reranker、向量库全部配置化，一键切换
- **Token 监控告警**：Agent 输入、检索结果超过阈值时自动告警，防止 prompt 溢出

## 技术栈

| 类别 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent / 工作流 | LangGraph + LangChain（ReAct Agent） |
| 向量数据库 | Milvus 3.0（BGE-M3 稠密+稀疏混合检索） |
| 对象存储 | MinIO |
| 对话历史 / 长期记忆 | MongoDB |
| 异步任务 | Celery + Redis |
| 文档解析 | MinerU（PDF → Markdown + 图片） |
| 多模态理解 | VL 模型（图片摘要生成） |
| LLM | DashScope / OpenAI 兼容接口 |
| 联网搜索 | MCP（Model Context Protocol） |
| 数据校验 | Pydantic + pydantic-settings |

## 项目结构

```
ThinkTank/
├── config/                  # 全局配置（LLM、Milvus、MinIO、Celery 等）
├── processor/               # 核心业务流水线
│   ├── import_processor/    # 文档导入流水线（7 个节点）
│   ├── query_processor/     # 检索问答流水线（8 个节点）
│   └── agent_processor/     # ReAct Agent（KB + Web 工具 + 状态管理）
├── services/                # 外部服务封装（Milvus、MinIO、MinerU 等）
├── utils/                   # 工具函数（Embedding、LLM、SSE、Celery 任务、长期记忆等）
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
git clone https://github.com/dongdongyang1/ThinkTank.git
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
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

支持 SSE 流式响应。`user_id` 用于长期记忆隔离，跨会话不变。

## 核心流程

### 导入流水线

```
PDF 上传 → MinerU 解析(PDF→MD) → 图片提取上传 MinIO → VL 模型图片摘要
→ 文档语义切分 → LLM 产品名识别 → BGE-M3 向量化 → 写入 Milvus
```

### ReAct Agent 问答流程

```
用户问题 → 加载长期记忆 → 构建历史对话 → Agent 决策
  ├─ 直接回答（闲聊/常识）
  ├─ 调用 KB Search → 混合检索 → 返回文档 → Agent 再决策
  └─ 调用 Web Search → MCP 联网搜索 → 返回结果 → Agent 再决策
     → 最多 5 轮循环 → 生成最终答案 → SSE 流式输出
     → 提取长期记忆 → 保存 MongoDB
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
