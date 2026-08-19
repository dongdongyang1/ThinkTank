import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional

import redis

# ===================== 全局常量 =====================
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

TASK_TTL_SECONDS = 7 * 24 * 3600  # 任务记录保留7天，防止Redis无限膨胀


# ===================== Redis 存储（进程间共享，重启不丢） =====================
# 1. 文件导入任务状态管理
_redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


def _task_key(task_id: str) -> str:
    """Redis key：每个任务一个hash"""
    return f"kb_import:task:{task_id}"


def init_task_record(task_id: str):
    """初始化任务记录（幂等：已存在则不覆盖，只续期）"""
    key = _task_key(task_id)
    if not _redis_client.exists(key):
        _redis_client.hset(key, mapping={
            "status": TASK_STATUS_PENDING,
            "running_nodes": json.dumps([]),
            "done_nodes": json.dumps([]),
            "result_cache": json.dumps({}),
            "create_time": str(time.time()),
        })
    _redis_client.expire(key, TASK_TTL_SECONDS)


def update_task_status(task_id: str, status: str, is_stream: bool = False):
    """更新任务整体状态"""
    init_task_record(task_id)
    _redis_client.hset(_task_key(task_id), "status", status)
    # 如果是流式任务，推送状态变更事件（SSE部分与原实现一致）
    if is_stream and task_id in sse_session_queues:
        from utils.sse_utils import SSEEvent
        push_to_session_nowait(task_id, SSEEvent.PROGRESS, {
            "status": status,
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id)
        })


def get_task_status(task_id: str) -> str:
    """获取任务全局状态（未知任务返回pending，与旧行为兼容）"""
    init_task_record(task_id)
    return _redis_client.hget(_task_key(task_id), "status")


def get_task_create_time(task_id: str) -> float:
    """获取任务创建时间戳（秒），供前端显示已运行时长"""
    init_task_record(task_id)
    return float(_redis_client.hget(_task_key(task_id), "create_time") or 0)


def _get_nodes(task_id: str, field: str) -> List[str]:
    init_task_record(task_id)
    raw = _redis_client.hget(_task_key(task_id), field)
    return json.loads(raw) if raw else []


def _set_nodes(task_id: str, field: str, nodes: List[str]):
    init_task_record(task_id)
    _redis_client.hset(_task_key(task_id), field, json.dumps(nodes))


def add_running_task(task_id: str, node_name: str):
    """标记节点正在运行"""
    running = _get_nodes(task_id, "running_nodes")
    if node_name not in running:
        running.append(node_name)
    _set_nodes(task_id, "running_nodes", running)


def add_done_task(task_id: str, node_name: str):
    """标记节点完成，移除运行列表，加入完成列表"""
    running = _get_nodes(task_id, "running_nodes")
    if node_name in running:
        running.remove(node_name)
    _set_nodes(task_id, "running_nodes", running)
    done = _get_nodes(task_id, "done_nodes")
    if node_name not in done:
        done.append(node_name)
    _set_nodes(task_id, "done_nodes", done)


def remove_running_task(task_id: str, node_name: str):
    """节点失败时移除运行标记（不加入完成列表）——修复3用"""
    running = _get_nodes(task_id, "running_nodes")
    if node_name in running:
        running.remove(node_name)
    _set_nodes(task_id, "running_nodes", running)


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行的节点列表"""
    return _get_nodes(task_id, "running_nodes")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表"""
    return _get_nodes(task_id, "done_nodes")


def set_task_result(task_id: str, key: str, value: Any):
    """存入任务结果/进度（修复2用，key如node_progress/error_msg）"""
    init_task_record(task_id)
    rk = _task_key(task_id)
    cache = json.loads(_redis_client.hget(rk, "result_cache") or "{}")
    cache[key] = value
    _redis_client.hset(rk, "result_cache", json.dumps(cache))


def get_task_result(task_id: str, key: str, default: Any = None) -> Any:
    """读取任务结果/进度"""
    init_task_record(task_id)
    cache = json.loads(_redis_client.hget(_task_key(task_id), "result_cache") or "{}")
    return cache.get(key, default)


def cleanup_task_record(task_id: str):
    """删除任务全部状态（修复原签名bug：多了个str参数）"""
    _redis_client.delete(_task_key(task_id))


# ===================== SSE会话队列管理（聊天流式，会话级数据保持进程内存，不改） =====================
sse_session_queues: Dict[str, asyncio.Queue] = {}
END_SIGNAL = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def create_sse_queue(session_id: str):
    """创建会话专属异步队列"""
    if session_id not in sse_session_queues:
        sse_session_queues[session_id] = asyncio.Queue()


async def push_to_session(session_id: str, event_type: str, data: dict):
    """向SSE会话推送事件消息"""
    if session_id not in sse_session_queues:
        create_sse_queue(session_id)
    queue = get_sse_queue(session_id)
    if not queue:
        return
    await queue.put({
        "event": event_type,
        "data": data
    })


def register_main_loop(loop: asyncio.AbstractEventLoop):
    """注册主事件循环（在 async 端点内用 asyncio.get_running_loop() 调用）"""
    global _main_loop
    _main_loop = loop


def push_to_session_nowait(session_id: str, event_type: str, data: dict):
    """线程安全的同步SSE推送：后台任务（线程池）中可安全调用"""
    if session_id not in sse_session_queues:
        create_sse_queue(session_id)
    queue = get_sse_queue(session_id)
    if not queue:
        return
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            push_to_session(session_id, event_type, data), _main_loop)
    else:
        sse_session_queues[session_id].put_nowait({
            "event": event_type, "data": data})


def get_sse_queue(session_id: str) -> Optional[asyncio.Queue]:
    """获取会话队列"""
    return sse_session_queues.get(session_id)


async def clear_sse_queue(session_id: str):
    """清空并删除会话队列"""
    if session_id in sse_session_queues:
        del sse_session_queues[session_id]

