import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI,HTTPException, Request, Header, Depends
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from starlette.templating import Jinja2Templates

from config.settings import settings
from processor.query_processor.logger import logger
from utils.celery_tasks.query_tasks import run_agent_task
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.task_utils import (
    get_task_status, get_task_result, get_new_deltas,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)

# 1. 创建应用
app = FastAPI(
    title="知识库问答-查询API",
    description="掌柜智库查询流程API（Agent + Celery 异步版）"
)

async def verify_api_key(x_api_key:str=Header(None,alias="X-API-Key"),api_key:str = ""):
    """API 鉴权：未配置 API_KEY 时跳过（开发环境），配置后强制校验"""
    key = x_api_key or api_key
    if settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401,detail="Invalid API Key")
    return key

# 2. 跨域（最小权限：显式列出方法和头部，来源从配置读取）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET","POST","DELETE"],
    allow_headers=["X-API-Key","Content-Type"],
)

# 3. 静态页面路由，服务端模板注入 API Key
templates = Jinja2Templates(directory=str(Path(__file__).absolute().parent.parent / "page"))

@app.get("/chat.html")
async def chat(request:Request):
    # current_dir_parent_path = Path(__file__).absolute().parent.parent
    # html_path = current_dir_parent_path / "page" / "chat.html"
    # if not html_path.exists():
    #     raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    # return FileResponse(html_path)
    return templates.TemplateResponse(request, "chat.html", {
        "api_key":settings.api_key,
    })


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="查询内容")
    session_id: Optional[str] = Field(None, max_length=128, description="会话ID")
    user_id: Optional[str] = Field(None, max_length=128, description="用户ID（前端生成，跨会话不变，用于长期记忆）")
    is_stream: bool = Field(False, description="是否流式返回")


def _sse_format(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def celery_sse_generator(task_id: str, request: Request):
    """
    轮询 Redis 的 SSE 生成器（Celery worker 在独立进程，通过 Redis 共享进度）
    - 每 0.1 秒轮询一次（降低延迟，提升流式体验）
    - 有新节点完成时推送 progress 事件
    - 有新 delta 时推送 delta 事件（流式输出）
    - 任务完成时推送 final 事件（含答案）
    - 任务失败时推送 error 事件
    """
    last_done_count = 0
    delta_index = 0
    max_wait_seconds = 300  # 最长等待5分钟，防止无限轮询
    waited = 0

    while True:
        # 客户端断开则停止
        if await request.is_disconnected():
            logger.info(f"SSE 客户端断开: task={task_id}")
            break

        status = get_task_status(task_id)
        progress = get_task_result(task_id, "progress", {})
        done_list = progress.get("done_list", [])

        # 有新节点完成时推送进度
        if len(done_list) > last_done_count:
            last_done_count = len(done_list)
            yield _sse_format("progress", {
                "done_list": done_list,
                "running_list": [],
                "status": status,
            })

        # 推送新的 delta（流式输出）
        new_deltas, delta_index = get_new_deltas(task_id, delta_index)
        for delta in new_deltas:
            yield _sse_format("delta", {"delta": delta})

        # 任务完成
        if status == TASK_STATUS_COMPLETED:
            answer = get_task_result(task_id, "answer", "")
            image_urls = get_task_result(task_id, "image_urls", [])
            # 最后再推一次剩余 delta
            remaining, delta_index = get_new_deltas(task_id, delta_index)
            for delta in remaining:
                yield _sse_format("delta", {"delta": delta})
            retrieved_contexts = get_task_result(task_id, "retrieved_contexts", [])
            yield _sse_format("final", {
                "answer": answer,
                "status": "completed",
                "image_urls": image_urls,
                "retrieved_contexts": retrieved_contexts,
            })
            logger.info(f"SSE 任务完成: task={task_id}, 答案长度={len(answer)}")
            break

        # 任务失败
        if status == TASK_STATUS_FAILED:
            error = get_task_result(task_id, "error", "未知错误")
            yield _sse_format("error", {"error": error})
            logger.error(f"SSE 任务失败: task={task_id}, error={error}")
            break

        # 超时保护
        if waited >= max_wait_seconds:
            yield _sse_format("error", {"error": "任务执行超时（超过5分钟）"})
            break

        waited += 0.1
        await asyncio.sleep(0.1)


@app.post("/query",dependencies=[Depends(verify_api_key)])
async def query(request: QueryRequest):
    """
    查询接口：提交 Celery 异步任务，立即返回 session_id + task_id
    - 流式：前端通过 /stream/{task_id} 轮询获取进度和结果
    - 非流式：服务端等待任务完成后返回结果
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    user_id = request.user_id if request.user_id else "default_user"
    task_id = str(uuid.uuid4())  # 每次请求唯一 task_id，避免同会话多任务 Redis 状态冲突
    is_stream = request.is_stream

    logger.info(f"提交查询任务: session={session_id}, user={user_id}, task={task_id}, stream={is_stream}, query={user_query}")

    # 提交 Celery 任务（异步，不阻塞）
    run_agent_task.delay(session_id, task_id, user_query, is_stream, user_id)

    if is_stream:
        # 流式：立即返回，前端连 /stream/{task_id} 轮询
        return {
            "message": "任务已提交，正在处理中...",
            "session_id": session_id,
            "task_id": task_id,
        }
    else:
        # 非流式：轮询 Redis 等待任务完成（最多等5分钟）
        max_wait = 300
        waited = 0
        while waited < max_wait:
            status = get_task_status(task_id)
            if status == TASK_STATUS_COMPLETED:
                answer = get_task_result(task_id, "answer", "")
                image_urls = get_task_result(task_id, "image_urls", [])
                retrieved_contexts = get_task_result(task_id, "retrieved_contexts", [])
                return {
                    "message": "处理完成！",
                    "session_id": session_id,
                    "task_id": task_id,
                    "answer": answer,
                    "image_urls": image_urls,
                    "retrieved_contexts": retrieved_contexts,
                    "done_list": [],
                }
            if status == TASK_STATUS_FAILED:
                error = get_task_result(task_id, "error", "未知错误")
                raise HTTPException(status_code=500, detail=f"任务执行失败: {error}")
            await asyncio.sleep(0.5)
            waited += 0.5

        raise HTTPException(status_code=504, detail="任务执行超时（超过5分钟）")


@app.get("/stream/{task_id}",dependencies=[Depends(verify_api_key)])
async def stream(task_id: str, request: Request):
    """SSE 实时返回结果（轮询 Redis，按 task_id 隔离）"""
    logger.info(f"SSE 连接建立: task={task_id}")
    return StreamingResponse(
        celery_sse_generator(task_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 历史会话管理
@app.delete("/history/{session_id}",dependencies=[Depends(verify_api_key)])
async def clear_chat_history(session_id: str):
    count = await asyncio.to_thread(clear_history, session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


@app.get("/history/{session_id}",dependencies=[Depends(verify_api_key)])
async def history(session_id: str, limit: int = 50):
    try:
        # 查看完整历史接口不限制时间窗口（max_age_hours=None），方便排查问题
        records = await asyncio.to_thread(get_recent_messages, session_id, limit=limit, max_age_hours=None)
        items = []
        for r in records:
            items.append({
                "id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.get("/health")
async def health():
    return {"ok": True}


from fastapi.responses import PlainTextResponse
@app.get("/favicon.ico")
async def favicon():
    return PlainTextResponse("", status_code=204)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
