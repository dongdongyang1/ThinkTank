import time
from typing import Optional, Callable

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from processor.agent_processor.agent_state import AgentState
from processor.agent_processor.prompt.react_prompt import build_system_prompt, MAX_LOOPS
from processor.agent_processor.tools.kb_search_tool import kb_search
from processor.agent_processor.tools.web_search_tool import web_search
from processor.query_processor.logger import logger
from utils.llm_utils import get_llm_client
from utils.text_cleaner import clean_markdown

load_dotenv()

# ==================== 常量配置 ====================
STREAM_CHUNK_DELAY = 0.02  # 流式推送每段延迟（秒），模拟打字机效果


class KBQueryAgent:
    """
    ReAct 知识库问答 Agent
    特性：流式输出 + 循环控制 + 长期记忆
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.tools = [kb_search, web_search]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # 流式 delta 推送回调（由 Celery 任务设置）
        # 流式输出的回调钩子，保存一个接收字符串、无返回的函数
        self.delta_callback: Optional[Callable[[str], None]] = None

        self.workflow = StateGraph(AgentState)
        self._register_nodes()
        self._setup_routes()
        self._compiled_app = None

    def set_delta_callback(self, callback: Callable[[str], None]):
        """设置流式 delta 推送回调"""
        self.delta_callback = callback

    def _register_nodes(self):
        self.workflow.add_node("agent", self._agent_node)
        self.workflow.add_node("tools", ToolNode(self.tools))

    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent 决策节点：支持流式输出最终答案"""
        loop_count = state.get("loop_count", 0) + 1
        logger.info(f"[agent node] 第 {loop_count} 轮决策，当前消息数: {len(state['messages'])}")

        # 构建系统提示词（注入长期记忆）
        system_prompt = build_system_prompt(state.get("long_term_memory", ""))
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

        # 达到循环上限时，强制不调用工具（在 prompt 中追加指令）
        if loop_count >= MAX_LOOPS:
            logger.warning(f"[agent node] 达到最大轮数 {MAX_LOOPS}，强制生成最终答案")
            messages[-1] = messages[-1]  # 保持原消息
            # 用一个不带工具的 LLM 生成最终答案
            response = self.llm.invoke(messages)
        else:
            # 流式调用，收集完整响应
            response = self._stream_llm_with_tools(messages, state.get("is_stream", False))

        tool_calls = response.tool_calls
        if tool_calls:
            tool_names = [tc["name"] for tc in tool_calls]
            logger.info(f"[agent node] 决策: 调用工具 {tool_names}")
        else:
            logger.info(f"[agent node] 决策: 直接回答（无工具调用）")

        return {
            "messages": [response],
            "loop_count": loop_count,
        }

    def _stream_llm_with_tools(self, messages, is_stream: bool) -> AIMessage:
        """
        流式调用绑定工具的 LLM。
        - 收集所有 chunk 合并为完整 AIMessage
        - 如果是最终答案（无 tool_calls）且开启流式，通过 delta_callback 推送
        """
        full_content = ""
        chunks = []

        for chunk in self.llm_with_tools.stream(messages):
            chunks.append(chunk)
            if chunk.content:
                full_content += chunk.content

        # 合并 chunk 为完整消息
        if chunks:
            response = chunks[0]
            for c in chunks[1:]:
                response += c
        else:
            response = AIMessage(content="")

        # 如果是最终答案（无 tool_calls）且开启流式，推送 delta
        if not response.tool_calls and is_stream and self.delta_callback and full_content:
            self._push_streaming_delta(full_content)

        return response

    def _push_streaming_delta(self, text: str):
        """
        模拟流式推送：清理 Markdown 后按句子切分，逐段推送 delta。
        前端收到 delta 后追加到答案区域，实现打字机效果。
        """
        if not self.delta_callback or not text:
            return

        # 先清理 Markdown，确保流式输出也是纯文本
        text = clean_markdown(text)
        if not text:
            return

        # 按句子/标点切分，避免切在半句话
        import re
        #分割：句号、感叹号、问号保留在每一段的末尾，不会被切掉丢掉。
        segments = re.split(r'(?<=[。！？.!?\n])', text)
        segments = [s for s in segments if s.strip()]

        for seg in segments:
            self.delta_callback(seg)
            time.sleep(STREAM_CHUNK_DELAY)

    def _route_after_agent(self, state: AgentState) -> str:
        """agent 节点后路由：有工具调用且未超限 → tools；否则 → END"""
        last_message = state["messages"][-1]
        loop_count = state.get("loop_count", 0)

        if last_message.tool_calls and loop_count < MAX_LOOPS:
            return "tools"
        return END

    def _setup_routes(self):
        self.workflow.set_entry_point("agent")
        self.workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", END: END}
        )
        self.workflow.add_edge("tools", "agent")

    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, state: AgentState, stream: bool = False):
        """统一执行入口"""
        if not self._compiled_app:
            self.compile()
            self._compiled_app.get_graph().print_ascii()

        if stream:
            return self._compiled_app.stream(state)
        return self._compiled_app.invoke(state)



