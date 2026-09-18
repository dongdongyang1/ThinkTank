# ThinkTank Docker 部署指南

## 一、文件清单

项目根目录下新增以下 5 个文件：

| 文件 | 作用 |
|------|------|
| `Dockerfile` | 项目镜像构建（Python + PyTorch + 依赖） |
| `docker-compose.yml` | 编排全部 9 个服务 |
| `.env.docker` | Docker 环境专用配置（服务名/容器内路径） |
| `requirements.txt` | Python 依赖清单 |
| `DEPLOY.md` | 本部署说明 |

---

## 二、部署架构

```
┌─────────────────────────────────────────────────────┐
│                   用户浏览器                           │
│            http://localhost:8002/chat.html           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Docker 网络（默认 bridge）                │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ query-api│  │import-api│  │  celery-worker   │  │
│  │  :8002   │  │  :8001   │  │  (GPU 必需)      │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                   │            │
│  ┌────▼──────────────▼───────────────────▼─────────┐ │
│  │              依赖服务容器                          │ │
│  │  Redis(:6379)  MongoDB(:27017)  Milvus(:19530) │ │
│  │  MinIO(:9002)   etcd(:2379)    milvus-minio     │ │
│  └──────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

---

## 三、前置准备

### 3.1 安装 Docker Desktop

1. 下载并安装 Docker Desktop：https://www.docker.com/products/docker-desktop/
2. 安装时勾选 "Use WSL 2 instead of Hyper-V"
3. 安装完成后启动 Docker Desktop，等待鲸鱼图标稳定

### 3.2 配置 WSL2 内存（重要！）

Milvus 容易 OOM 崩溃，必须给 WSL2 分配足够内存：

1. 按 `Win + R`，输入 `%UserProfile%`，回车
2. 新建文件 `.wslconfig`，内容：
```ini
[wsl2]
memory=6GB
processors=4
swap=2GB
```
3. 关闭 Docker Desktop，命令行执行 `wsl --shutdown`
4. 重新打开 Docker Desktop

### 3.3 GPU 支持（可选，没有 GPU 可跳过）

BGE-M3 嵌入模型需要 GPU，没有 GPU 也能用但很慢。

1. 安装 NVIDIA 显卡驱动（最新版）
2. 安装 nvidia-container-toolkit：
   - 下载：https://github.com/NVIDIA/nvidia-container-toolkit/releases
   - 安装后重启 Docker Desktop
3. 验证：`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### 3.4 准备模型文件

把本地模型文件复制到项目目录下的 `models/` 文件夹：

```
项目根目录/
├── models/
│   ├── BAAI--bge-m3/
│   │   └── snapshots/
│   │       └── master/          ← BGE-M3 模型
│   └── bge-reranker-large/      ← BGE 重排序模型（可选，用 qwen3-rerank 时不需要）
```

> 模型文件从本地 `D:\modelscope_cache\models\` 复制过来即可。

---

## 四、部署步骤

### 第1步：修改配置（如需要）

打开 `.env.docker`，检查以下配置：

- `BGE_DEVICE=cuda:0` — 没有 GPU 改成 `cpu`
- `MINERU_API_TOKEN` — 确认是你的有效 token
- `OPENAI_API_KEY` — 确认是你的有效 API Key

其他配置（Milvus/MongoDB/Redis/MinIO 地址）已经改成 Docker 服务名，**不要改**。

### 第2步：构建项目镜像

在项目根目录打开命令行，执行：

```bash
docker compose build
```

> 首次构建需要 5~15 分钟（下载基础镜像 + 安装依赖），后续构建有缓存会很快。

### 第3步：启动所有服务

```bash
docker compose up -d
```

> `-d` 表示后台运行。首次启动 Milvus 需要 1~2 分钟初始化。

### 第4步：查看服务状态

```bash
docker compose ps
```

所有服务状态应该是 `running` 或 `healthy`。

如果某个服务没起来，查看日志：

```bash
docker compose logs -f celery-worker
docker compose logs -f milvus-standalone
```

### 第5步：验证

1. **查询 API**：浏览器打开 http://localhost:8002/health，返回 `{"ok":true}`
2. **导入 API**：浏览器打开 http://localhost:8001/docs，看到 Swagger 文档
3. **聊天页面**：浏览器打开 http://localhost:8002/chat.html
4. **MinIO 控制台**：浏览器打开 http://localhost:9001，账号 admin / Admin@123456
5. **Milvus**：用 Attu 连接 localhost:19530，能看到 kb_chunks 和 kb_item_names 集合

---

## 五、常用操作

### 停止所有服务
```bash
docker compose down
```

### 重启某个服务
```bash
docker compose restart celery-worker
```

### 重新构建并启动（代码修改后）
```bash
docker compose build && docker compose up -d
```

### 查看实时日志
```bash
docker compose logs -f query-api
docker compose logs -f celery-worker
```

### 进入容器调试
```bash
docker exec -it thinktank-celery-worker bash
```

### 清空所有数据（慎用！）
```bash
docker compose down -v
```

---

## 六、注意事项

### 6.1 端口冲突

如果本地已经跑了 Redis/MongoDB/Milvus/MinIO，会端口冲突。两种解决方式：
1. 停掉本地的服务，用 Docker 里的
2. 修改 `docker-compose.yml` 里的端口映射（比如把 `6379:6379` 改成 `6380:6379`）

### 6.2 没有 GPU

如果没有 GPU：
1. `.env.docker` 里 `BGE_DEVICE=cpu`
2. `docker-compose.yml` 里删掉 `celery-worker` 的 `deploy` 段（GPU 配置）
3. 嵌入速度会慢很多，建议导入小文档测试

### 6.3 数据持久化

所有数据存在 Docker volume 里：
- MongoDB 数据：`mongo-data` volume
- Milvus 数据：`milvus-db` volume
- MinIO 数据：`minio-data` volume
- Redis 数据：`redis-data` volume

`docker compose down` 不会删数据，`docker compose down -v` 才会删。

### 6.4 模型文件

模型文件通过 `./models:/models` 挂载到容器，不打进镜像。更换模型只需要替换本地 `models/` 目录，重启容器即可。

### 6.5 外部 API

MinerU、DashScope LLM、qwen3-rerank、MCP WebSearch 都是外部 HTTP 服务，容器内能访问外网即可。如果公司网络有代理，需要配置 Docker 代理。

---

## 七、故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| Milvus 启动后又挂了 | WSL2 内存不足 | 按 3.2 调大 WSL2 内存到 6GB |
| celery-worker 启动失败 | GPU 没配置好 | 按 3.3 安装 nvidia-container-toolkit，或改用 CPU |
| 连接不上 Milvus | Milvus 还在初始化 | 等 1~2 分钟，或看日志 `docker compose logs milvus-standalone` |
| 导入 PDF 失败 | MinerU token 无效 | 检查 `.env.docker` 里的 MINERU_API_TOKEN |
| 聊天没反应 | celery-worker 没起来 | `docker compose ps` 查看状态，`docker compose logs celery-worker` 看日志 |
| 端口被占用 | 本地有相同服务 | 停掉本地服务，或修改 docker-compose.yml 端口映射 |

---

## 八、卸载

```bash
# 停止并删除容器（保留数据）
docker compose down

# 停止并删除容器 + 数据（全部清空）
docker compose down -v

# 删除项目镜像
docker rmi thinktank-query-api thinktank-import-api thinktank-celery-worker
```
