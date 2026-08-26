import re
import uuid

from langchain_core.tools import tool

from processor.query_processor.logger import logger
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp


@tool
def web_search(query: str):
    """
    联网搜索互联网信息。
    仅在以下情况使用：
    1. search_kb 两次检索均未命中或结果明显不足
    2. 用户明确要求"上网搜""查一下最新信息"
    3. 询问时效性信息（如最新价格、固件版本）且知识库未覆盖

    【重要】产品相关问题必须先调用 search_kb，禁止跳过知识库直接联网。

    Args:
        query: 搜索关键词

    Returns:
        联网搜索到的网页内容摘要，包含标题、链接和内容
    """
    # 参数校验：空值直接返回
    query = (query or "").strip()
    if not query:
        return "查询内容为空，请提供具体的产品问题。"
    # 参数校验：超长截断（防止 LLM 生成超长 query 导致搜索 API 报错）
    if len(query) > 500:
        query = query[:500]
    # 参数校验：压缩连续空白
    query = re.sub(r"\s+", " ", query).strip()

    try:
        logger.info(f"[web_search tool] 开始搜索: {query}")

        # 使用唯一临时 session_id，避免共享历史
        temp_session_id = f"web_search_{uuid.uuid4().hex[:8]}"

        temp_state = {
            "original_query": query,
            "session_id": temp_session_id,
            "is_stream": False,
            "web_search_docs": [],
            "item_names": [],
            "rewritten_query": query,  # 必须设为 query，否则 NodeWebSearchMcp 取到空串会跳过搜索
            "message_id": "",
        }

        web_node = NodeWebSearchMcp()
        temp_state = web_node(temp_state)

        docs = temp_state.get("web_search_docs", [])
        if not docs:
            return "联网搜索未找到相关信息。"

        formatted_parts = []
        for idx, doc in enumerate(docs, start=1):
            content = doc.get("content", "") or doc.get("text", "")
            title = doc.get("title", "无标题")
            url = doc.get("url", "")
            formatted_parts.append(
                f"[网页{idx}] 标题: {title} | 链接: {url}\n内容: {content}"
            )
        result = "\n\n---\n\n".join(formatted_parts)
        logger.info(f"[web_search tool] 搜索完成，返回 {len(docs)} 条结果")
        return result
    except Exception as e:
        logger.error(f"[web_search tool] 执行失败: {e}", exc_info=True)
        return "联网搜索暂时不可用，请稍后重试或仅基于知识库回答。"
