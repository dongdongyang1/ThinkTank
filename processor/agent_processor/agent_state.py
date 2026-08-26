from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Agent 工作记忆状态
    - messages: 对话历史 + Agent 思考过程 + 工具调用结果（ReAct 核心载体）
    - session_id: 会话标识
    - is_stream: 是否流式输出
    - query_intent: LLM 判断的查询意图
    - rewritten_query: LLM 改写后的规范查询词
    - retrieved_docs: 检索到的文档
    - loop_count: 当前循环轮数（agent→tools→agent 为一轮）
    - reflection_count: 反思次数
    - long_term_memory: 长期记忆内容（注入系统提示词）
    """
    #Annotated[类型, 归约函数] 把新消息追加到老列表后面
    messages : Annotated[list[BaseMessage], add_messages]
    session_id : str
    is_stream : bool
    query_intent : str
    rewritten_query : str
    retrieved_docs : list[dict]
    loop_count : int
    reflection_count : int
    long_term_memory : str
