import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.logger import logger
from processor.query_processor.state import QueryGraphState
from utils.json_format_utils import format_json


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    name : str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        logger.info(f"【{self.name}】节点逻辑")
        query = state.get("rewritten_query","")
        docs = []
        # 如果没有查询内容，直接返回
        if query:
            #asyncio.run专门用来启动、运行异步协程
            result = asyncio.run(self._mcp_call(query))
            if result:
                logger.info(f"MCP原始返回result:{result}")
                text = result.content[0].text
                logger.info(f"text原始内容 >>>{text}<<<")
                data = json.loads(text) if isinstance(text, str) else text
                pages = data.get("pages") or []
                # 统一输出结构化结果，供后续 rerank/引用使用
                # 每条：{title, url, snippet}
                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({"title":title,"url":url,"snippet":snippet})
                    logger.info(f"MCP 搜索结果:{docs}")
        if docs:
            state["web_search_docs"] = docs
            return state
        # 无搜索结果也必须返回状态更新；返回 None 会导致 langgraph 报错
        return {"web_search_docs": []}

    async def _mcp_call(self,query:str):
        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers":{"Authorization":f"Bearer {mcp_config.api_key}"},
                "timeout":10
            },
            cache_tools_list=True,
            max_retry_attempts=3
        )
        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query":query,"count":5}
            )
            return result
        finally:
            await search_mcp.cleanup()


if __name__ == "__main__":
    init_state = {
        "rewritten_query":"关于brother HAK180烫金机，如何调节转印温度？"
    }
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(format_json(result))
