import re
import uuid

from langchain_core.tools import tool

from config.settings import settings
from processor.query_processor.logger import logger
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_hyde import NodeSearchHyde
from utils.mongo_history_utils import clear_history


def _extract_model_keyword(name: str) -> str:
    """
    从商品名提取最具体最长型号标识，如 '华为擎云W585X' -> 'W585X'，无型号返回空串
    'H3C LA2608室内无线网关' -> 'LA2608'（而非'H3C'）
    """
    toks = sorted(set(re.findall(r"[A-Za-z]+[\-]?\d+[A-Za-z]*",name)),key=len,reverse=True)
    return toks[0] if toks else ""


def _isolate_single_model(query: str, item_names: list, is_comparison: bool = False,
                          all_item_names: list = None) -> list:
    """
    单型号问题隔离：商品名确认可能因宽泛关键词（如"擎云"）补充进多个易混淆型号，
    导致检索混入其他型号、Agent 张冠李戴。此函数在单型号问题时，
    只保留 item_name 中型号标识出现在用户问题里的那个型号。
    同时做"query 型号补全"：query 明确提到的型号若被 LLM/规则漏掉（含只提取了 1 个错误型号的情况），
    从商品名全表补回，保证召回率。
    是否对比类问题优先用 LLM 判定（is_comparison），关键词规则作兜底。
    """
    # LLM 判定为对比类问题：需要多个型号做对比，不隔离
    if is_comparison:
        logger.info(f"[kb_search] LLM判定为对比类问题，保留全部型号: {item_names}")
        return item_names

    # 关键词兜底：LLM 未判定为对比，但 query 明显含对比词时也保留全部
    comparison_markers = ["和", "与", "对比", "区别", "分别", "哪个", "不同", "vs", "versus", "还是"]
    if any(m in query for m in comparison_markers):
        logger.info(f"[kb_search] 关键词兜底判定为对比类问题，保留全部型号: {item_names}")
        return item_names

    # 提取 query 中出现的型号（精确匹配，避免 W585 误配 W585X）
    query_clean = query.replace(" ", "").replace("\n", "")
    query_models = set(re.findall(r"[A-Za-z]+[\-]?\d+[A-Za-z]*", query_clean))

    # query 中无明确型号标识：无法判断型号归属，保守保留原 item_names（不隔离、不误杀）
    if not query_models:
        logger.info(f"[kb_search] query 无明确型号，保留原 item_names: {item_names}")
        return item_names

    # 单型号问题：
    # 1. 保留 item_names 中型号 ∈ query_models 的（过滤无关型号，如 B530/W525）
    kept = [n for n in item_names if _extract_model_keyword(n) in query_models]

    # 2. query 型号补全：query 提到的型号在 item_names 里缺失时，从全表补回（保证召回率）
    if all_item_names:
        kept_models = {_extract_model_keyword(n) for n in kept}
        for qm in query_models:
            if qm in kept_models:
                continue
            for full_name in all_item_names:
                if _extract_model_keyword(full_name) == qm:
                    kept.append(full_name)
                    kept_models.add(qm)
                    logger.info(f"[kb_search] 补全 query 型号 {qm}: 补入 {full_name}")
                    break

    if kept:
        logger.info(f"[kb_search] 单型号隔离(含补全): {item_names} -> {kept}")
        return kept

    # 3. 兜底：query 型号在 item_names 和全表都未找到，保守保留原 item_names
    logger.info(f"[kb_search] 未匹配到 query 型号，保留原 item_names: {item_names}")
    return item_names

def _search_for_comparison(query:str,item_names:list) ->str:
    """
    对比问题专用：拆成两个子问题，分别检索，按型号分组返回
    """
    if len(item_names)<2:
        return None  # 不是对比问题，走原来的流程

    # 拆成两个子问题
    results_by_item = {}
    for item_name in item_names:
        # 构造子查询：只问这个型号的问题
        sub_query = f"{item_name} {query}"

        # 执行完整检索流程（产品名确认 → HyDE → 向量检索 → RRF → Rerank）
        temp_state = {
            "original_query": sub_query,
            "session_id": f"compare_{uuid.uuid4().hex[:8]}",
            "is_stream": False,
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "web_search_docs": [],
            "rrf_chunks": [],
            "reranked_docs": [],
            "item_names": [item_name],
            "rewritten_query": sub_query,
            "history": [],
            "answer": "",
            "message_id": "",
            "prompt": "",
            "is_summary": False,
        }

        #HyDE
        hyde_node = NodeSearchHyde()
        temp_state.update(hyde_node(temp_state))

        #向量检索
        emb_node = NodeSearchEmbedding()
        temp_state.update(emb_node(temp_state))

        #RRF
        rrf_node = NodeRrf()
        temp_state.update(rrf_node(temp_state))

        #Rerank
        rerank_node = NodeRerank()
        temp_state.update(rerank_node(temp_state))

        results_by_item[item_name] = temp_state.get("reranked_docs",[])

    # 按型号分组格式化
    parts = []
    for item_name,docs in results_by_item.items():
        parts.append(f"【{item_name}】的文档")
        for idx,doc in enumerate(docs[:5],start=1):
            parts.append(f"[文档{idx}] 标题: {doc.get('title', '')}\n内容: {doc.get('content', '')}")
        parts.append("")
    return "\n\n---\n\n".join(parts)
def _balance_docs_by_item(docs, per_item=3, max_total=8):
    """按商品平衡截断：每个 item_name 各保留分数最高的 per_item 条，防止某产品占满全部名额"""
    groups = {}
    for d in docs:
        groups.setdefault(d.get("item_name", ""), []).append(d)
    picked = []

    for g in groups.values():
        g.sort(key=lambda x: x.get("score") or 0, reverse=True)
        picked.extend(g[:per_item])
    picked.sort(key=lambda x: x.get("score") or 0, reverse=True)
    return picked[:max_total]


@tool
def kb_search(query : str) -> str:
    """
    检索本地产品知识库，返回相关文档片段。
    知识库包含：产品手册、规格参数、操作指南、安全注意事项、故障排查等。
    当用户询问产品相关问题时，必须优先使用此工具。

    Args:
        query: 检索关键词或问题描述（建议使用规范表述，避免口语化）

    Returns:
        格式化后的知识库文档片段，包含标题、来源、相关度和内容
    """
    # 参数校验：空值直接返回
    query = (query or "").strip()
    if not query:
        return "查询内容为空，请提供具体的产品问题。"
    # 参数校验：超长截断（防止 LLM 生成超长 query 导致检索 API 报错）
    if len(query) > 500:
        query = query[:500]
    # 参数校验：压缩连续空白
    query = re.sub(r"\s+", " ", query).strip()

    try:
        logger.info(f"[search_kb tool] 开始检索: {query}")

        # 使用唯一临时 session_id，避免所有工具调用共享同一份历史
        # （NodeItemNameConfirm 内部会读写 MongoDB 历史，共享 id 会导致商品名提取混乱）
        temp_session_id = f"kb_search_{uuid.uuid4().hex[:8]}"

        temp_state = {
            "original_query": query,
            "session_id": temp_session_id,
            "is_stream": False,
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "web_search_docs": [],  # 保持空，本工具不做联网搜索
            "rrf_chunks": [],
            "reranked_docs": [],
            "item_names": [],
            "rewritten_query": "",
            "history": [],
            "answer": "",
            "message_id": "",
            "prompt": "",
            "is_summary": False,
        }

        # 1. 物品名确认（核心作用：提取 item_names 用于检索过滤）
        item_node = NodeItemNameConfirm()
        temp_state.update(item_node(temp_state))

        #关键词兜底
        comparison_markers = ["和", "与", "对比", "区别", "分别", "哪个", "不同", "vs", "versus", "还是"]
        is_comparison_by_keyword = any(m in query for m in comparison_markers)

        # 只要 LLM 说对比问题，或者 query 里有对比关键词，且 item_names >= 2，就算对比问题
        if (temp_state.get("is_comparison") or is_comparison_by_keyword) and len(temp_state.get("item_names", [])) >= 2:
            logger.info(
                f"[kb_search] 检测到对比问题，拆分为多个子问题分别检索 (LLM={temp_state.get('is_comparison')}, 关键词={is_comparison_by_keyword})")
            comparison_result = _search_for_comparison(
                query,
                temp_state["item_names"]
            )
            if comparison_result:
                return comparison_result

        # 清理临时 session 的历史记录：NodeItemNameConfirm 内部会写 MongoDB，
        # 但 kb_search 用的是临时 session_id，这些记录永远不会被读取，直接删掉避免垃圾数据累积
        clear_history(temp_session_id)

        #拒答/确认短路：NodeItemNameConfirm 分支B(候选反问)/分支C(产品未找到) 已给出答复
        if temp_state.get("answer"):
            if not temp_state.get("item_names") and settings.UNKNOWN_PRODUCT_BLOCK_WEB:
                # 分支C + 评估模式：禁止 agent 转 web_search 硬答
                return f"【内部标记】allow_web_search=false\n\n{temp_state['answer']}"
            return temp_state["answer"]

        # 单型号问题隔离：防止宽泛关键词（如"擎云"）把同一系列多个易混淆型号都带进来，
        # 导致检索混入其他型号、Agent 把别的型号功能安到问题型号上。
        if temp_state.get("item_names"):
            is_comparison = temp_state.get("is_comparison", False)
            # 传入商品名全表，用于 query 型号补全（保证召回率）
            try:
                all_item_names = item_node._load_all_item_names()
            except Exception as e:
                logger.warning(f"[kb_search] 加载商品名全表失败: {e}")
                all_item_names = None
            temp_state["item_names"] = _isolate_single_model(
                query, temp_state["item_names"], is_comparison, all_item_names)
            logger.info(f"[kb_search] 隔离后 item_names: {temp_state['item_names']}")

        # 强制重置查询字段：NodeItemNameConfirm 可能把 original_query/rewritten_query 搞丢或设为 None，
        # 导致后续 HyDE 和 Rerank 节点 query 为空，API 报 "query should not be empty"。
        # 这里直接用传入的 query，不依赖上游节点的改写结果。
        # temp_state["original_query"] = query
        # temp_state["rewritten_query"] = query
        #logger.info(f"[kb_search] 重置查询字段: original_query={query[:50]}, rewritten_query={query[:50]}")
        rw = (temp_state.get("rewritten_query") or "").strip()
        temp_state["original_query"] = query
        temp_state["rewritten_query"] = rw if len(rw) >= 4 else query
        logger.info(f"[kb_search] 查询字段: original={query[:50]}, rewritten={temp_state['rewritten_query'][:50]}")

        # 2. 两路并行检索（embedding + HyDE)
        # 注意：必须用 update 合并结果，不能直接赋值替换！
        # 节点的 process() 只返回自己的字段（如 {"embedding_chunks": [...]}），
        # 直接赋值会导致 original_query/rewritten_query/item_names 等字段丢失。
        emb_node = NodeSearchEmbedding()
        hyde_node = NodeSearchHyde()
        temp_state.update(emb_node(temp_state))
        temp_state.update(hyde_node(temp_state))

        # 3. RRF融合（两路结果）
        rrf_node = NodeRrf()
        temp_state.update(rrf_node(temp_state))

        # 4. Rerank重排
        rerank_node = NodeRerank()
        temp_state.update(rerank_node(temp_state))
        if temp_state.get("is_comparison") or len(temp_state.get("item_names", [])) >= 2:
            temp_state["reranked_docs"] = _balance_docs_by_item(temp_state.get("reranked_docs", []))



        # 5. 格式化结果
        reranked_docs = temp_state.get("reranked_docs", [])

        # 从意图识别结果中取出联网权限标记
        allow_web_search = temp_state.get("allow_web_search",True)
        if not reranked_docs:
            if not allow_web_search:
                return "未检索到相关文档。当前问题涉及内部产品信息，不允许联网搜索，请调整问题后重试。"
            return "未检索到相关文档。建议尝试其他关键词，或使用 web_search 联网搜索。"

        formatted_parts = []
        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content", "")
            title = doc.get("title", "无标题")
            source = doc.get("source", "local")
            item_name = doc.get("item_name", "未知商品")
            # 注意：doc.get("score", 0) 在 score 为 None 时仍返回 None，
            # 后续 f"{score:.4f}" 会报 unsupported format string passed to NoneType.__format__
            score = doc.get("score")
            if score is None:
                score = 0.0
            formatted_parts.append(f"[文档{idx}] 标题: {title} | 来源: {source} | 商品: {item_name} | 相关度: {score:.4f}\n"
                f"内容: {content}")
        result = "\n\n---\n\n".join(formatted_parts)

        # ===== token 检测 =====
        from utils.token_utils import estimate_tokens
        result_tok = estimate_tokens(result)
        doc_toks = [estimate_tokens(doc.get("content", "")) for doc in reranked_docs]

        web_flag = f"【内部标记】allow_web_search={'true' if allow_web_search else 'false'}"
        result = f"{web_flag}\n\n{result}"

        logger.info(f"[search_kb tool] 检索完成，返回 {len(reranked_docs)} 条文档,总token={result_tok}，各文档token={doc_toks},allow_web_search={allow_web_search}")

        if result_tok > settings.KB_SEARCH_WARN_TOKENS:
            logger.warning(
                f"[search_kb tool] 单次检索返回超过{settings.KB_SEARCH_WARN_TOKENS}token({result_tok})，文档数={len(reranked_docs)}")
        return result
    except Exception as e:
        logger.error(f"[search_kb tool] 执行失败: {e}", exc_info=True)
        return "知识库检索暂时不可用，请稍后重试或尝试联网搜索。"
