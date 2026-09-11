"""
知识库查询 Celery 任务（增强版）
特性：流式输出 + 反思修正 + 长期记忆 + 循环控制
Agent 执行放到独立 worker 进程，不阻塞 FastAPI 事件循环。
进度和结果通过 Redis 共享，SSE 端点轮询 Redis 推送给前端。
"""
import re
import traceback
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config.celery_config import celery
from processor.agent_processor.agent_graph import KBQueryAgent
from processor.query_processor.logger import logger
from utils.long_term_memory import get_long_term_memory
from utils.mongo_history_utils import save_chat_messages, get_recent_messages
from utils.task_utils import (
    set_task_result, update_task_status, push_delta,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)
from utils.text_cleaner import clean_markdown

# Agent 单例（worker 进程内复用，避免每次重新编译图）
_agent = None

# 历史对话最大轮数（user+assistant 算一轮）——从最新往最老加，受 token 预算约束
HISTORY_MAX_TURNS = 8
# 历史区总 token 预算（user 全保留；assistant 截断；超预算从最老开始丢）
HISTORY_MAX_TOKENS = 4000
# assistant 历史回答注入时最多保留字符数（只留开头结论，避免长文淹没当前问题）
ASSISTANT_TRUNCATE_CHARS = 120
# assistant 回答超过该长度时，生成一句话摘要（上下文压缩）；更短的直接原样存
ASSISTANT_SUMMARIZE_MIN_CHARS = 200
# 摘要提示词：把长回答提炼成一句话结论，保留型号
SUMMARIZE_TEMPLATE = """用一句话概括下面回答的核心结论（不超过50字），保留涉及的产品型号，不要多余解释：
{answer}"""

# 匹配 Markdown 图片语法 ![alt](url)，提取 url（不限制扩展名，MinIO 图片 URL 都提取）
_MD_IMAGE_PATTERN = re.compile(r'!\[.*?\]\((https?://.+?\.(?:jpg|jpeg|png|gif|webp|bmp))\)', re.IGNORECASE)


def _get_agent():
    global _agent
    if _agent is None:
        _agent = KBQueryAgent()
    return _agent


def _build_agent_messages(session_id: str, user_query: str) -> list:
    """
    构建 Agent 的消息列表：历史对话 + 当前用户消息。
    - 分角色：user 全保留；assistant 截断（只留开头结论，避免重读长文淹没当前问题）
    - token 预算：从最新往最老累计，超 HISTORY_MAX_TOKENS 即停止
    """
    messages = []
    try:
        history = get_recent_messages(session_id, limit=HISTORY_MAX_TURNS * 2)
        # 从最新往最老处理，便于按预算丢弃最旧消息
        budget_chars = HISTORY_MAX_TOKENS * 2  # 保守估 1 token ≈ 2 字符
        used_chars = 0
        for msg in reversed(history):
            role = msg.get("role", "")
            text = msg.get("text", "")
            if not text:
                continue
            if role == "user":
                content = text
            elif role == "assistant":
                # 上下文压缩：优先用存库时生成的一句话摘要；没有摘要则硬截断兜底
                content = msg.get("summary") or _truncate_assistant(text, ASSISTANT_TRUNCATE_CHARS)
            else:
                continue
            if used_chars + len(content) > budget_chars:
                break  # 已到预算上限，更老的直接丢弃
            used_chars += len(content)
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.reverse()  # 还原为时间正序
        from utils.token_utils import estimate_tokens
        history_tok = sum(estimate_tokens(m.content) for m in messages if hasattr(m,"content"))
        logger.info(f"[Celery] 加载历史对话 {len(messages)} 条, 历史区≈{history_tok} tokens (字符数={used_chars})")
    except Exception as e:
        logger.error(f"[Celery] 加载历史对话失败: {e}")

    messages.append(HumanMessage(content=user_query))
    return messages


def _truncate_assistant(text: str, max_chars: int) -> str:
    """assistant 历史回答截断：去掉旧图片 URL + 压缩空白，只保留开头结论，限长。"""
    text = _MD_IMAGE_PATTERN.sub("", text)  # 旧图片 URL 注入新 prompt 无意义，去掉
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _summarize_answer(answer: str) -> str:
    """用 LLM 把长回答提炼成一句话摘要（qwen-flash，上下文压缩），失败时退回截断。"""
    try:
        from utils.llm_utils import get_llm_client
        from config.lm_config import lm_config
        llm = get_llm_client(model=lm_config.item_model)
        prompt = SUMMARIZE_TEMPLATE.format(answer=(answer or "")[:1500])
        summary = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        if summary:
            return summary[:120]  # 摘要本身也限制长度，防止 LLM 输出过长
        return _truncate_assistant(answer, ASSISTANT_TRUNCATE_CHARS)
    except Exception as e:
        logger.error(f"[Celery] 生成回答摘要失败: {e}")
        return _truncate_assistant(answer, ASSISTANT_TRUNCATE_CHARS)


def _extract_images_from_messages(messages: list) -> list:
    """从 Agent 的所有 ToolMessage 中提取图片 URL，去重后返回。"""
    image_urls = []
    seen = set()
    tool_msg_count = 0
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_msg_count += 1
        content = msg.content or ""
        matches = _MD_IMAGE_PATTERN.findall(content)
        logger.info(f"[Celery] ToolMessage[{tool_msg_count}] 长度={len(content)}, 匹配到图片={len(matches)}")
        for match in matches:
            url = match.strip()
            # 兜底：对URL路径中的中文/特殊字符做编码，处理旧数据未编码的情况
            if url:
                from urllib.parse import urlparse, urlunparse, quote
                parsed = urlparse(url)
                if parsed.path:
                    encoded_path = quote(parsed.path, safe='/')
                    url = urlunparse(parsed._replace(path=encoded_path))
            if url and url not in seen:
                seen.add(url)
                image_urls.append(url)
    logger.info(f"[Celery] 共 {tool_msg_count} 条 ToolMessage，提取到 {len(image_urls)} 张图片")
    if image_urls:
        for i, url in enumerate(image_urls):
            logger.info(f"[Celery] 图片[{i+1}]: {url}")
    return image_urls


def _make_delta_callback(task_id: str):
    """创建 delta 推送回调（闭包绑定 task_id）"""
    def callback(delta: str):
        push_delta(task_id, delta)
    return callback


@celery.task(bind=True, name="query.run_agent", max_retries=0)
def run_agent_task(self, session_id: str, task_id: str, user_query: str, is_stream: bool = False, user_id: str = "default_user"):
    """
    Celery 任务：执行 Agent 查询流程（增强版）
    - session_id：MongoDB 历史对话的隔离键（每次新对话变化）
    - user_id：长期记忆的隔离键（前端生成，跨会话不变）
    - task_id：Redis 任务状态隔离（每次请求唯一）
    """
    logger.info(f"[Celery] run_agent_task 开始: session={session_id}, user={user_id}, task={task_id}, query={user_query}")

    # 1. 加载长期记忆（按 user_id 隔离，跨会话生效）
    long_term_memory = ""
    try:
        ltm = get_long_term_memory()
        long_term_memory = ltm.format_memories_for_prompt(user_id, min_importance=3)
        if long_term_memory:
            logger.info(f"[Celery] 加载长期记忆:\n{long_term_memory}")
    except Exception as e:
        logger.error(f"[Celery] 加载长期记忆失败: {e}")

    # 2. 构建带历史对话的消息列表
    agent_messages = _build_agent_messages(session_id, user_query)

    # 3. 初始化状态
    init_state = {
        "messages": agent_messages,
        "session_id": session_id,
        "user_id": user_id,
        "is_stream": is_stream,
        "query_intent": "",
        "rewritten_query": "",
        "retrieved_docs": [],
        "loop_count": 0,
        "reflection_count": 0,
        "long_term_memory": long_term_memory,
    }

    try:
        agent = _get_agent()

        # 设置流式 delta 回调
        if is_stream:
            agent.set_delta_callback(_make_delta_callback(task_id))
        else:
            agent.set_delta_callback(None)

        update_task_status(task_id, TASK_STATUS_PROCESSING, is_stream=False)

        # 4. 执行 Agent 图（流式逐节点记录进度）
        # 注意：LangGraph stream 返回的是每个节点的状态增量，需要自己合并成完整状态
        from langgraph.graph.message import add_messages
        full_state = dict(init_state)  # 从初始状态开始
        done_list = []
        for event in agent.run(init_state, stream=True):
            for node_name, node_state in event.items():
                # 合并状态增量到完整状态
                for key, value in node_state.items():
                    if key == "messages":
                        full_state["messages"] = add_messages(
                            full_state.get("messages", []), value
                        )
                    else:
                        full_state[key] = value
                done_list.append(node_name)
                set_task_result(task_id, "progress", {
                    "done_list": list(done_list),
                    "running_list": [],
                    "status": TASK_STATUS_PROCESSING,
                })

        # 5. 取最终答案并清理 Markdown（兜底，确保纯文本输出）
        messages = full_state.get("messages", [])
        answer = messages[-1].content if messages else ""
        answer = clean_markdown(answer)

        # ===== token 检测：最终 messages 峰值 =====
        from utils.token_utils import estimate_messages_tokens, format_token_report
        final_tok = estimate_messages_tokens(messages)
        logger.info(format_token_report(final_tok, prefix="[Celery] 最终状态峰值 | "))

        loop_count = full_state.get("loop_count", 0)
        reflection_count = full_state.get("reflection_count", 0)

        # 6. 提取图片
        image_urls = _extract_images_from_messages(messages)

        # 7. 最终结果写 Redis
        set_task_result(task_id, "answer", answer)
        set_task_result(task_id, "image_urls", image_urls)
        update_task_status(task_id, TASK_STATUS_COMPLETED, is_stream=False)

        # 8. 写入 MongoDB 历史
        save_chat_messages(
            session_id=session_id, role="user", text=user_query,
            rewritten_query="", item_names=[], image_urls=image_urls,
        )
        if answer:
            # 上下文压缩：长回答生成一句话摘要存库（qwen-flash，收尾阶段用户无感）
            summary = _summarize_answer(answer) if len(answer) > ASSISTANT_SUMMARIZE_MIN_CHARS else answer
            save_chat_messages(
                session_id=session_id, role="assistant", text=answer,
                rewritten_query="", item_names=[], image_urls=image_urls,
                summary=summary,
            )

        # 9. 提取长期记忆（异步，不阻塞主流程）：LLM 提取 + 超量淘汰
        try:
            ltm = get_long_term_memory()
            ltm.extract_and_save(user_id, user_query, answer)
            ltm.prune_user(user_id)
        except Exception as e:
            logger.error(f"[Celery] 提取长期记忆失败: {e}")

        logger.info(
            f"[Celery] run_agent_task 完成: task={task_id}, "
            f"答案长度={len(answer)}, 循环={loop_count}, "
            f"图片={len(image_urls)}"
        )
        return {
            "status": "completed",
            "answer_length": len(answer),
            "loop_count": loop_count,
            "reflection_count": reflection_count,
            "image_count": len(image_urls),
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[Celery] run_agent_task 异常: task={task_id}, {e}\n{tb}")
        set_task_result(task_id, "error", str(e))
        set_task_result(task_id, "traceback", tb)
        update_task_status(task_id, TASK_STATUS_FAILED, is_stream=False)
        return {"status": "failed", "error": str(e)}
