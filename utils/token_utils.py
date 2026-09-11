import re
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

def estimate_tokens(text: str) -> int:
    """
    轻量 token 估算（中文为主场景）。
    规则：中文字符 1字≈0.7token，英文/数字 1字≈0.25token，其他符号折中。
    精度足够做监控和告警，不用于精确计费。
    """
    if not text:
        return 0
    # 区分中文字符和非中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text)) #\u 是 Python 字符串的 Unicode 转义符号
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 0.7 + other_chars * 0.25)


def estimate_messages_tokens(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    统计消息列表的 token 分布。
    返回：总 token、各角色 token、各角色消息数、ToolMessage 详情。
    """
    role_stats = {}
    tool_details = []
    total_tokens = 0

    for i, msg in enumerate(messages):
        role = _get_role(msg)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        tok = estimate_tokens(content)
        total_tokens += tok

        if role not in role_stats:
            role_stats[role] = {"count": 0, "tokens": 0}
        role_stats[role]["count"] += 1
        role_stats[role]["tokens"] += tok

        if role == "tool":
            tool_details.append({"index": i, "tokens": tok, "preview": content[:60]})

    return {
        "total_tokens": total_tokens,
        "total_messages": len(messages),
        "role_stats": role_stats,
        "tool_details": tool_details,
    }


def _get_role(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool"
    return type(msg).__name__ # 都不是时，返回类名


def format_token_report(stats: Dict[str, Any], prefix: str = "") -> str:
    """格式化为一行日志，方便 grep"""
    parts = [f"{prefix}总token={stats['total_tokens']}", f"消息数={stats['total_messages']}"]
    for role, s in stats["role_stats"].items():
        parts.append(f"{role}:{s['count']}条/{s['tokens']}tok")
    if stats["tool_details"]:
        tool_toks = [t["tokens"] for t in stats["tool_details"]]
        parts.append(f"ToolMessage分别={tool_toks}")
    return " | ".join(parts)
