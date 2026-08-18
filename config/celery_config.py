from celery import Celery

redis_url = "redis://127.0.0.1:6379/0"

celery = Celery(
    "kb_import",
    broker=redis_url,
    backend=redis_url
)

# 自动注册任务
celery.autodiscover_tasks(["processor.celery_tasks"])