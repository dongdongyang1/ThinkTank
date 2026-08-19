import os
import sys
from pathlib import Path

from celery import Celery

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

celery = Celery(
    "kb_import",
    broker=redis_url,
    backend=redis_url
)

# 自动注册任务
celery.autodiscover_tasks(["utils.celery_tasks"])

# 任务执行参数加固
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    broker_connection_retry_on_startup=True,  # 启动时Redis未就绪则重试连接，不直接崩
    task_time_limit=2700,  # 硬超时45分钟：超时强杀，防止僵尸任务占着worker
    task_soft_time_limit=2400,  # 软超时40分钟：抛SoftTimeLimitError，先于硬超时
    worker_prefetch_multiplier=1,  # 每个worker同时只预取1个任务
    worker_max_tasks_per_child=50,  # 每worker执行50个任务后自动重启，防内存泄漏
)

