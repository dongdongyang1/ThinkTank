import asyncio
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_utils import SSEEvent, create_sse_stream
from utils.task_utils import (
    create_sse_queue, update_task_status, TASK_STATUS_PROCESSING,
    get_task_result, set_task_result, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    push_to_session_nowait, register_main_loop, push_to_session,
)

# 1. 创建应用
app = FastAPI(
    title="知识库问答-查询API",
    description="此文档是掌柜智库查询流程的API接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)

# 3. 静态页面路由
@app.get("/chat.html")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path/ "page" / "chat.html"

    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query : str = Field(..., description="查询内容")
    session_id : Optional[str] = Field(None,description="会话ID")
    is_stream : bool  = Field(False,description="是否流式返回")


@app.post("/query")
async def query(background_tasks:BackgroundTasks,request:QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 注册主事件循环，供后台任务（线程池）跨线程推送SSE
    register_main_loop(asyncio.get_running_loop())

    # 处理是不是流式返回结果
    is_stream = request.is_stream
    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)
    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph,session_id,user_query,is_stream)
        # 返回结果
        print("开始处理结果....")
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        # 同步运行：用 to_thread 避免阻塞事件循环
        # （node_web_search_mcp 内部有 asyncio.run，不能直接在 loop 线程里执行）
        await asyncio.to_thread(run_query_graph, session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        return {
            "message": "处理完成！",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }


# 定义查询接口
def run_query_graph(session_id:str,user_query:str,is_stream:bool=False):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")
    init_state = {
        "original_query":user_query,
        "session_id":session_id,
        "is_stream":is_stream
    }

    try:
        workflow = KBQueryWorkflow()
        if is_stream:
            # langgraph 的 stream() 是惰性生成器，必须迭代才会真正执行图
            final_state = {}
            done_list = []
            for event in workflow.run(init_state, stream=True):
                for node_name, node_state in event.items():
                    final_state = node_state
                    done_list.append(node_name)
                    push_to_session_nowait(session_id, SSEEvent.PROGRESS, {
                        "done_list": list(done_list),
                        "running_list": [],
                        "status": TASK_STATUS_PROCESSING,
                    })
            answer = final_state.get("answer", "") if isinstance(final_state, dict) else ""
            update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
            # 前端收到 final 事件后主动 closeSSE（chat.html:296-300）
            push_to_session_nowait(session_id, SSEEvent.FINAL, {"answer": answer})
        else:
            final_state = workflow.run(init_state, stream=False)
            update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
            # 把答案写入结果缓存，供 /query 同步分支读取
            set_task_result(session_id, "answer", final_state.get("answer", ""))

    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)

        if is_stream:
            push_to_session_nowait(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.get("/stream/{session_id}")
async def stream(session_id:str,request:Request):
    """
    sse 实时返回结果
    """
    print("调用流式/stream...")
    return create_sse_stream(session_id,request)


#历史会话管理
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id:str):
    """
    清空指定会话的历史记录
    """
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


@app.get("/history/{session_id}")
async def history(session_id:str,limit:int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id,limit=limit)
        items = []
        for r in records:
            items.append({
                "id" : str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id" : r.get("session_id",""),
                "role" : r.get("role",""),
                "text" : r.get("text",""),
                "rewritten_query" : r.get("rewritten_query",""),
                "item_names" : r.get("item_names",[]),
                "ts" : r.get("ts")
            })
        return {"session_id":session_id,"items":items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}


from fastapi.responses import PlainTextResponse
@app.get("/favicon.ico")
async def favicon():
    return PlainTextResponse("", status_code=204)


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8002)
