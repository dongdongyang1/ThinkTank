import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from enum import StrEnum
from utils.task_utils import get_sse_queue, END_SIGNAL, clear_sse_queue, cleanup_task_record


# SSE事件类型枚举，和文档完全对应
class SSEEvent(StrEnum):
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"

async def sse_generator(session_id: str, request: Request):
    """
    标准SSE生成器，匹配文档 /stream/{session_id} 接口
    """
    queue = get_sse_queue(session_id)
    if not queue:
        yield f"event: {SSEEvent.ERROR}\ndata: {json.dumps({'error': '会话不存在'})}\n\n"
        return

    try:
        while True:
            msg = await queue.get()
            if msg == END_SIGNAL:
                break
            event_name = msg["event"]
            data = json.dumps(msg["data"], ensure_ascii=False)
            # 标准SSE格式输出
            yield f"event: {event_name}\n"
            yield f"data: {data}\n\n"
    except Exception as e:
        err_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"event: {SSEEvent.ERROR}\ndata: {err_data}\n\n"
    finally:
        await clear_sse_queue(session_id)
        cleanup_task_record(session_id)

def create_sse_stream(session_id: str, request: Request) -> StreamingResponse:
    """快速构造SSE流式响应"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )