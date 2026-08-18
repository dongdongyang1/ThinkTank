import asyncio
from typing import Dict, List, Any, Optional

# ===================== 全局常量 =====================
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# ===================== 全局内存存储 =====================
# 1. 文件导入任务状态管理（task_id 为key）
task_global_status: Dict[str, str] = {}
task_running_nodes: Dict[str, List[str]] = {}  # 正在执行的节点
task_done_nodes: Dict[str, List[str]] = {}     # 已完成节点
task_result_cache: Dict[str, Dict[str, Any]] = {}  # 任务结果缓存

# 2. SSE会话队列（query聊天流式专用，session_id为key）
sse_session_queues: Dict[str, asyncio.Queue] = {}
END_SIGNAL = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ===================== 文件导入任务相关方法 =====================
def init_task_record(task_id: str):
    """初始化任务记录"""
    if task_id not in task_global_status:
        task_global_status[task_id] = TASK_STATUS_PENDING
        task_running_nodes[task_id] = []
        task_done_nodes[task_id] = []
        task_result_cache[task_id] = {}

def update_task_status(task_id: str, status: str, is_stream: bool = False):
    """更新任务整体状态"""
    init_task_record(task_id)
    task_global_status[task_id] = status
    # 如果是流式任务，推送状态变更事件
    if is_stream and task_id in sse_session_queues:
        from utils.sse_utils import SSEEvent
        push_to_session_nowait(task_id, SSEEvent.PROGRESS, {
            "status": status,
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id)
        })

def get_task_status(task_id: str) -> str:
    """获取任务全局状态"""
    init_task_record(task_id)
    return task_global_status[task_id]

def add_running_task(task_id: str, node_name: str):
    """标记节点正在运行"""
    init_task_record(task_id)
    if node_name not in task_running_nodes[task_id]:
        task_running_nodes[task_id].append(node_name)

def add_done_task(task_id: str, node_name: str):
    """标记节点完成，移除运行列表，加入完成列表"""
    init_task_record(task_id)
    if node_name in task_running_nodes[task_id]:
        task_running_nodes[task_id].remove(node_name)
    if node_name not in task_done_nodes[task_id]:
        task_done_nodes[task_id].append(node_name)

def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行的节点列表"""
    init_task_record(task_id)
    return task_running_nodes[task_id]

def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表"""
    init_task_record(task_id)
    return task_done_nodes[task_id]

def set_task_result(task_id: str, key: str, value: Any):
    """存入任务结果"""
    init_task_record(task_id)
    task_result_cache[task_id][key] = value

def get_task_result(task_id: str, key: str, default: Any = None) -> Any:
    """读取任务结果"""
    init_task_record(task_id)
    return task_result_cache[task_id].get(key, default)

# ===================== SSE会话队列管理（聊天流式） =====================
def create_sse_queue(session_id: str):
    """创建会话专属异步队列"""
    if session_id not in sse_session_queues:
        sse_session_queues[session_id] = asyncio.Queue()

async def push_to_session(session_id: str, event_type: str, data: dict):
    """向SSE会话推送事件消息"""
    if session_id not in sse_session_queues:
        create_sse_queue(session_id)
    queue = sse_session_queues[session_id]
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
    if _main_loop and _main_loop.is_running():
        # 协程提交到主循环执行，安全唤醒正在 await queue.get() 的消费者
        asyncio.run_coroutine_threadsafe(
             push_to_session(session_id, event_type, data), _main_loop)
    else:
        # 无运行中循环（脚本/测试环境）时兜底
        sse_session_queues[session_id].put_nowait({
            "event": event_type, "data": data})


def get_sse_queue(session_id: str) -> Optional[asyncio.Queue]:
    """获取会话队列"""
    return sse_session_queues.get(session_id)

async def clear_sse_queue(session_id: str):
    """清空并删除会话队列"""
    if session_id in sse_session_queues:
        del sse_session_queues[session_id]