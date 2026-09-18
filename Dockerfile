# ==================== 基础镜像（支持 GPU） ====================
# 如果没有 GPU，可改成 python:3.11-slim，并把 BGE_DEVICE 设为 cpu
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ==================== 系统依赖 ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==================== Python 依赖 ====================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 卸载 torchaudio / torchvision（项目不需要音频/视频处理，且基础镜像预装版本与 transformers 不兼容）
RUN pip uninstall -y torchaudio torchvision || true

# ==================== 复制项目代码 ====================
COPY . .

# ==================== 环境变量（模型离线模式） ====================
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PYTHONUNBUFFERED=1

# ==================== 暴露端口 ====================
# 8001: 导入API  8002: 查询API
EXPOSE 8001 8002

# ==================== 默认启动命令（被 docker-compose 各服务覆盖） ====================
CMD ["uvicorn", "web.api.query_service:app", "--host", "0.0.0.0", "--port", "8002"]
