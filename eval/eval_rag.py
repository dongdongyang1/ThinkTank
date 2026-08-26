"""
ThinkTank RAG 系统评估脚本（v2）
==================================
核心改动（相对 v1）：
  1. 启动时从 Milvus 全量加载一次 file_title -> item_name 映射（只查一次），
     将标注的 relevant_doc_ids（填的是 file_title）转换为 item_name 集合。
  2. 检索指标统一用 item_name 匹配（与检索过滤逻辑一致），
     算 Recall@K / Precision@K / F1 / MRR。
  3. 新增内容级相关性判断：用 qwen-flash 逐个判断检索到的 chunk 是否真的对
     回答问题有帮助，算内容级 Recall / Precision / MRR，并与文档级 item_name
     匹配结果做对比。
  4. 生成指标用 LLM-as-Judge 一次调用算 Faithfulness / Answer Relevancy /
     Context Precision / Context Recall（JSON 输出，减少 API 调用）。
  5. 结果保存为 JSON，控制台打印汇总报告（含文档级 vs 内容级对比）。

用法：
  python eval_rag.py                       # 评估全部样本
  python eval_rag.py --limit 10            # 只评估前10条
  python eval_rag.py --confusion-only      # 只评估混淆测试10条(TT-014~TT-023)
  python eval_rag.py --skip-gen             # 跳生成指标，只算检索指标
  python eval_rag.py --skip-content-judge   # 跳内容级 LLM 判断，只算文档级
  python eval_rag.py --top-k 5              # 指定 Top-K
"""

import os
import sys
import json
import re
import time
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

# ==================== 路径配置 ====================
PROJECT_ROOT = r"D:\ThinkTank"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 飞书表格配置
FEISHU_URL = "https://my.feishu.cn/sheets/UJZEsFaSyhI0uJtzUWdcAqM7nYg"
SHEET_NAME = "评估集标注表"

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "../eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM 评估模型
JUDGE_MODEL = "qwen-flash"


# ====================================================================
# 1. 从飞书表读取评估样本
# ====================================================================
def load_eval_samples(
    limit: int = None,
    confusion_only: bool = False,
    from_json: str = None,
) -> List[Dict]:
    """
    从飞书表格读取评估样本；若 lark-cli 不可用或指定了 from_json，则从本地 JSON 加载。
    返回列表，每个元素包含：sample_id, category, question, question_type,
    ground_truth, relevant_doc_ids (file_title 列表), relevant_keywords, difficulty, notes
    """
    print("[1/6] 加载评估样本...")

    # 优先从本地 JSON 加载（显式指定或 lark-cli 不可用时自动 fallback）
    if from_json:
        samples = _load_from_json(from_json)
    else:
        try:
            samples = _load_from_feishu()
        except FileNotFoundError:
            print(f"  lark-cli 不在 PATH 中，自动降级到本地 JSON")
            samples = _load_from_local_json_files()

    if not samples:
        print("  错误: 未能加载任何评估样本")
        sys.exit(1)

    # 筛选
    if confusion_only:
        samples = [s for s in samples if s["sample_id"] in (f"TT-{i:03d}" for i in range(14, 24))]
    if limit:
        samples = samples[:limit]

    print(f"  已加载 {len(samples)} 条评估样本")
    return samples


def _load_from_feishu() -> List[Dict]:
    """通过 lark-cli 从飞书表格读取评估样本"""
    result = subprocess.run(
        ["lark-cli", "sheets", "+csv-get",
         "--url", FEISHU_URL,
         "--sheet-name", SHEET_NAME,
         "--range", "A1:K51"],
        capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"  错误: lark-cli 调用失败: {result.stderr}")
        sys.exit(1)

    data = json.loads(result.stdout)
    csv_text = data["data"]["annotated_csv"]

    samples = []
    headers = None
    for line in csv_text.strip().split("\n"):
        if line.startswith("[row="):
            line = line.split("]", 1)[1].strip()
        parts = _parse_csv_line(line)
        if headers is None:
            headers = parts
            continue
        if len(parts) < 6:
            continue
        sample = dict(zip(headers, parts))
        sid = sample.get("样本ID", "")
        if not sid or sid in ("TT-001", "TT-002", "TT-003"):
            continue
        samples.append(_build_sample_dict(sample))

    return samples


def _load_from_json(filepath: str) -> List[Dict]:
    """
    从本地 JSON 文件加载评估样本。
    支持两种格式：
      1. 飞书导出的二维数组格式：[[{value:..}, {value:..}], ...]
      2. 已解析的字典列表格式：[{sample_id:.., question:..}, ...]
    列顺序（飞书 A-K）：0=样本ID, 1=知识领域, 2=用户问题, 3=问题类型,
      4=标准答案, 5=相关文档ID, 6=检索关键词, 7=难度, 8=备注
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 格式2：已是字典列表
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "sample_id" in raw[0]:
        return raw

    # 格式1：二维数组（飞书导出）
    samples = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 6:
            continue
        # 提取每个单元格的 value
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                cells.append(str(cell.get("value", "")).strip())
            else:
                cells.append(str(cell).strip())

        sid = cells[0] if len(cells) > 0 else ""
        # 跳过空行和示例行
        if not sid or sid in ("TT-001", "TT-002", "TT-003"):
            continue

        sample = {
            "样本ID": sid,
            "知识领域": cells[1] if len(cells) > 1 else "",
            "用户问题": cells[2] if len(cells) > 2 else "",
            "问题类型": cells[3] if len(cells) > 3 else "",
            "标准答案": cells[4] if len(cells) > 4 else "",
            "相关文档ID（;分隔）": cells[5] if len(cells) > 5 else "",
            "检索关键词（;分隔）": cells[6] if len(cells) > 6 else "",
            "难度": cells[7] if len(cells) > 7 else "",
            "备注": cells[8] if len(cells) > 8 else "",
        }
        samples.append(_build_sample_dict(sample))

    return samples


def _load_from_local_json_files() -> List[Dict]:
    """
    自动查找并合并项目目录下所有 eval_samples*.json 文件（按文件名排序）。
    用于 lark-cli 不可用时的 fallback。
    """
    import glob
    pattern = os.path.join(PROJECT_ROOT, "eval_samples*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  错误: 未找到任何 eval_samples*.json 文件（pattern: {pattern}）")
        sys.exit(1)

    all_samples = []
    seen_ids = set()
    for fpath in files:
        print(f"  加载本地文件: {os.path.basename(fpath)}")
        batch = _load_from_json(fpath)
        for s in batch:
            sid = s["sample_id"]
            if sid not in seen_ids:
                all_samples.append(s)
                seen_ids.add(sid)
    # 按 sample_id 排序
    all_samples.sort(key=lambda s: s["sample_id"])
    return all_samples


def _build_sample_dict(sample: Dict[str, str]) -> Dict:
    """将原始字段字典转换为统一的样本字典格式"""
    return {
        "sample_id": sample.get("样本ID", ""),
        "category": sample.get("知识领域", ""),
        "question": sample.get("用户问题", ""),
        "question_type": sample.get("问题类型", ""),
        "ground_truth": sample.get("标准答案", ""),
        "relevant_doc_ids": [d.strip() for d in sample.get("相关文档ID（;分隔）", "").split(";") if d.strip()],
        "relevant_keywords": sample.get("检索关键词（;分隔）", ""),
        "difficulty": sample.get("难度", ""),
        "notes": sample.get("备注", ""),
    }


def _parse_csv_line(line: str) -> List[str]:
    """解析 CSV 行，处理带引号的字段"""
    result = []
    current = ""
    in_quotes = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                current += '"'
                i += 2
                continue
            in_quotes = not in_quotes
        elif c == "," and not in_quotes:
            result.append(current)
            current = ""
        else:
            current += c
        i += 1
    result.append(current)
    return result


# ====================================================================
# 2. Milvus 全量映射加载（只查一次）
# ====================================================================
def load_milvus_mappings() -> Tuple[Dict[str, Set[str]], Dict[str, int], Dict[Any, str], int]:
    """
    启动时从 Milvus 全量查询一次，建立三个映射：
      1. file_title -> set(item_name)   （核心：标注 file_title 转 item_name）
      2. item_name -> chunk_count        （内容级 Recall 的分母）
      3. chunks_id  -> item_name         （给 reranked_docs 补 item_name）
    返回 (file_to_items, item_chunk_count, chunk_to_item, total_chunks)
    """
    print("[2/6] 从 Milvus 全量加载 file_title -> item_name 映射（仅一次）...")

    from utils.milvus_utils import get_milvus_client
    from config.milvus_config import milvus_config

    client = get_milvus_client()
    collection_name = milvus_config.chunks_collection

    file_to_items: Dict[str, Set[str]] = defaultdict(set)
    item_chunk_count: Dict[str, int] = defaultdict(int)
    chunk_to_item: Dict[Any, str] = {}

    # 分页全量查询（3626 条，小数据量，一次 limit=10000 足够；保留分页逻辑应对增长）
    offset = 0
    batch_size = 2000
    total = 0

    while True:
        batch = client.query(
            collection_name=collection_name,
            output_fields=["chunks_id", "file_title", "item_name"],
            limit=batch_size,
            offset=offset,
        )
        if not batch:
            break
        for r in batch:
            ft = (r.get("file_title") or "").strip()
            it = (r.get("item_name") or "").strip()
            cid = r.get("chunks_id")
            if ft:
                file_to_items[ft].add(it)
            if it:
                item_chunk_count[it] += 1
            if cid is not None:
                chunk_to_item[cid] = it
        total += len(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    # 统计信息
    unique_files = len(file_to_items)
    unique_items = len(item_chunk_count)
    print(f"  全量 chunk 数: {total}")
    print(f"  唯一 file_title: {unique_files}, 唯一 item_name: {unique_items}")

    # 打印映射概览（前 10 条）
    print("  映射概览（file_title -> item_name）:")
    for i, (ft, items) in enumerate(sorted(file_to_items.items())):
        if i >= 10:
            print(f"    ... 共 {unique_files} 条")
            break
        print(f"    {ft} -> {items}")

    return dict(file_to_items), dict(item_chunk_count), chunk_to_item, total


def convert_relevant_to_item_names(
    relevant_doc_ids: List[str],
    file_to_items: Dict[str, Set[str]],
    sample_id: str = "",
) -> Set[str]:
    """
    将标注的 relevant_doc_ids（file_title 列表）转换为 item_name 集合。
    找不到映射的 file_title 会打印警告并跳过。
    """
    item_names: Set[str] = set()
    missing = []
    for ft in relevant_doc_ids:
        if ft in file_to_items:
            item_names.update(file_to_items[ft])
        else:
            missing.append(ft)
    if missing:
        print(f"    警告 [{sample_id}]: 以下 file_title 在 Milvus 中无映射，已跳过: {missing}")
    # 过滤空字符串
    item_names.discard("")
    return item_names


# ====================================================================
# 3. 调用 Query 流程
# ====================================================================
def run_query_workflow(question: str, session_id: str) -> Dict[str, Any]:
    """
    调用 ThinkTank Query 流程。
    返回：{reranked_docs, rrf_chunks, answer, embedding_chunks, hyde_embedding_chunks,
           item_names, rewritten_query, error}
    """
    from processor.query_processor.main_graph import KBQueryWorkflow

    workflow = KBQueryWorkflow()
    init_state = {
        "original_query": question,
        "session_id": session_id,
        "is_stream": False,
    }

    try:
        final_state = workflow.run(init_state, stream=False)
        return {
            "reranked_docs": final_state.get("reranked_docs", []),
            "rrf_chunks": final_state.get("rrf_chunks", []),
            "answer": final_state.get("answer", ""),
            "embedding_chunks": final_state.get("embedding_chunks", []),
            "hyde_embedding_chunks": final_state.get("hyde_embedding_chunks", []),
            "item_names": final_state.get("item_names", []),
            "rewritten_query": final_state.get("rewritten_query", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "reranked_docs": [],
            "rrf_chunks": [],
            "answer": "",
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "item_names": [],
            "rewritten_query": "",
            "error": str(e),
        }


def enrich_reranked_docs_with_item_name(
    reranked_docs: List[Dict],
    chunk_to_item: Dict[Any, str],
) -> List[Dict]:
    """
    给 reranked_docs 补回 item_name 字段。
    rerank 节点合并时丢弃了 item_name，这里用全量 chunks_id->item_name 映射补回。
    web 搜索结果（chunks_id=None, source="web"）标记 item_name=""。
    """
    for doc in reranked_docs:
        cid = doc.get("chunks_id")
        source = doc.get("source", "local")
        if source == "web" or cid is None:
            doc["item_name"] = ""
        else:
            doc["item_name"] = chunk_to_item.get(cid, "")
    return reranked_docs


# ====================================================================
# 4. 文档级检索指标（item_name 匹配）
# ====================================================================
def calc_doc_level_metrics(
    reranked_docs: List[Dict],
    relevant_item_names: Set[str],
    k: int = 5,
) -> Dict[str, Any]:
    """
    文档级检索指标：用 item_name 匹配（与检索过滤逻辑一致）。
    只统计 source="local" 且有 item_name 的文档；web 结果不计入文档级指标。

    返回 recall_at_k, precision_at_k, f1, mrr, num_hits, num_relevant,
    num_retrieved_local, first_rel_rank, hits, missed, retrieved_item_names
    """
    # 从 local 文档提取 item_name，按排名顺序去重
    seen = set()
    retrieved_items: List[str] = []
    local_count = 0
    for doc in reranked_docs:
        if doc.get("source") == "web":
            continue
        local_count += 1
        it = (doc.get("item_name") or "").strip()
        if it and it not in seen:
            retrieved_items.append(it)
            seen.add(it)

    # 取 Top-K（按去重后的 item_name 排名）
    top_k_items = retrieved_items[:k]
    retrieved_set = set(top_k_items)

    hits = retrieved_set & relevant_item_names
    num_hits = len(hits)

    recall = num_hits / len(relevant_item_names) if relevant_item_names else 0.0
    precision = num_hits / len(top_k_items) if top_k_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MRR：第一个命中的 item_name 排名的倒数
    mrr = 0.0
    first_rel_rank = None
    for rank, it in enumerate(top_k_items, 1):
        if it in relevant_item_names:
            mrr = 1.0 / rank
            first_rel_rank = rank
            break

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "num_hits": num_hits,
        "num_relevant": len(relevant_item_names),
        "num_retrieved_local": local_count,
        "num_unique_items": len(retrieved_items),
        "first_rel_rank": first_rel_rank,
        "hits": sorted(hits),
        "missed": sorted(relevant_item_names - retrieved_set),
        "retrieved_item_names": retrieved_items,
    }


# ====================================================================
# 5. 内容级相关性判断（LLM qwen-flash）
# ====================================================================
def judge_chunks_relevance_batch(
    question: str,
    chunks: List[Dict],
) -> Dict[int, int]:
    """
    批量判断检索到的 chunk 是否对回答问题有实质性帮助。
    一次 LLM 调用判断所有 chunk，返回 {chunk_index(1-based): 0或1}。

    判断标准：
      1（相关）：片段包含能直接回答问题的关键信息、操作步骤、参数规格或事实依据
      0（不相关）：与问题无关，或仅含泛泛安全警告/版权声明/目录导航等无实质帮助内容
    """
    if not chunks:
        return {}

    # 构造每个 chunk 的描述（截断长文本）
    chunk_descs = []
    for i, chunk in enumerate(chunks, 1):
        content = (chunk.get("content") or "").strip()[:800]
        source = chunk.get("source", "local")
        item_name = (chunk.get("item_name") or "").strip()
        if item_name:
            prefix = f"[{i}] (来源:{source}, 商品:{item_name}) "
        else:
            prefix = f"[{i}] (来源:{source}) "
        chunk_descs.append(prefix + content)

    chunks_text = "\n\n".join(chunk_descs)

    prompt = f"""你是一个RAG系统评估专家。请判断以下每个检索片段是否对回答用户问题有实质性帮助。

判断标准：
- 1（相关）：片段包含能直接回答问题的关键信息、操作步骤、参数规格或事实依据
- 0（不相关）：片段与问题无关，或仅包含泛泛的安全警告、版权声明、目录导航等无实质帮助的内容

【用户问题】
{question}

【检索片段】
{chunks_text}

请以JSON格式返回每个片段的判断结果，key为片段编号（字符串），value为0或1。
例如：{{"1": 1, "2": 0, "3": 1}}
只返回JSON，不要其他内容。"""

    try:
        from utils.llm_utils import get_llm_client
        client = get_llm_client(model=JUDGE_MODEL, json_mode=True)
        response = client.invoke([{"role": "user", "content": prompt}])
        result_text = response.content.strip()
        result = json.loads(result_text)
        return {int(k): int(v) for k, v in result.items()}
    except Exception as e:
        print(f"    内容相关性 LLM 判断失败: {e}")
        # 降级：全部视为不相关
        return {i: 0 for i in range(1, len(chunks) + 1)}


def calc_content_level_metrics(
    reranked_docs: List[Dict],
    relevance_judgments: Dict[int, int],
    relevant_item_names: Set[str],
    item_chunk_count: Dict[str, int],
    k: int = 5,
) -> Dict[str, Any]:
    """
    内容级检索指标：基于 LLM 对每个 chunk 的相关性判断。

    - Content Precision@K：Top-K 中 LLM 判断为相关的 chunk 比例（含 web）
    - Content Recall：检索到且 LLM 判断为相关的 local chunk 数 / 标注相关商品总 chunk 数
      （分母从全量映射 item_chunk_count 统计，近似估计）
    - Content MRR：第一个 LLM 判断为相关的 chunk 排名倒数
    - Content Hit Rate@K：Top-K 中至少有一个相关 chunk 的比例（单条为 0/1）
    """
    total_chunks = len(reranked_docs)
    top_k = reranked_docs[:k]

    # 统计 LLM 判断为相关的 chunk
    relevant_in_topk = 0
    relevant_in_all = 0
    relevant_local_in_all = 0  # 仅 local 且相关
    first_rel_rank = None

    for i, doc in enumerate(reranked_docs, 1):
        is_rel = relevance_judgments.get(i, 0) == 1
        if is_rel:
            relevant_in_all += 1
            if doc.get("source") != "web":
                relevant_local_in_all += 1
            if first_rel_rank is None:
                first_rel_rank = i
        if i <= k and is_rel:
            relevant_in_topk += 1

    # Precision@K
    precision = relevant_in_topk / k if k > 0 else 0.0

    # Recall（近似）：分母 = 标注相关 item_name 下的总 chunk 数
    denom = sum(item_chunk_count.get(it, 0) for it in relevant_item_names)
    recall = relevant_local_in_all / denom if denom > 0 else 0.0

    # MRR
    mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0

    # Hit Rate
    hit_rate = 1.0 if relevant_in_topk > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "recall_approx": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "hit_rate_at_k": round(hit_rate, 4),
        "relevant_in_topk": relevant_in_topk,
        "relevant_in_all": relevant_in_all,
        "relevant_local_in_all": relevant_local_in_all,
        "total_retrieved": total_chunks,
        "denominator_chunks": denom,
        "first_rel_rank": first_rel_rank,
        "judgments": relevance_judgments,
    }


# ====================================================================
# 6. 生成指标（LLM-as-Judge，一次调用算 4 项）
# ====================================================================
def calc_generation_metrics(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    relevant_item_names: Set[str],
) -> Dict[str, float]:
    """
    一次 LLM 调用计算 4 项生成指标（JSON 输出）：
      - faithfulness: 答案中的事实声明是否都能在上下文中找到依据（无幻觉）
      - answer_relevancy: 答案是否直接、完整地回答了用户问题
      - context_precision: 检索上下文中与问题真正相关的比例
      - context_recall: 参考答案的关键信息点有多少能在上下文中找到
    """
    if not answer:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    context_text = "\n---\n".join((c or "")[:1000] for c in contexts[:5])
    items_str = ", ".join(sorted(relevant_item_names)) if relevant_item_names else "（无）"

    prompt = f"""你是一个RAG系统评估专家。请从四个维度评估以下问答对，每项给0到1之间的小数分数。

【用户问题】
{question}

【标注的相关商品】
{items_str}

【参考答案】
{ground_truth}

【检索上下文（Top-5）】
{context_text}

【待评估答案】
{answer}

【四个维度的评分标准】

1. faithfulness（忠实度）：答案中的每一个事实声明是否都能在检索上下文中找到依据。
   - 1.0：完全基于上下文，无任何幻觉
   - 0.8：基本基于上下文，极少量无关补充
   - 0.5：部分基于上下文，存在明显未经验证的信息
   - 0.2：大部分编造，与上下文关联弱
   - 0.0：完全编造

2. answer_relevancy（答案相关性）：答案是否直接、完整地回答了用户问题，是否答非所问或遗漏关键信息。
   - 1.0：完整准确回答，与参考答案高度一致
   - 0.8：基本回答，少量细节遗漏
   - 0.5：部分回答，明显遗漏或跑题
   - 0.2：仅略微涉及，大部分无关
   - 0.0：完全答非所问

3. context_precision（上下文精确率）：检索到的上下文中，有多少比例与用户问题真正相关。
   - 1.0：所有上下文都高度相关
   - 0.8：大部分相关，少量不相关
   - 0.5：约一半相关
   - 0.2：大部分不相关
   - 0.0：全部不相关

4. context_recall（上下文召回率）：参考答案中的关键信息点，有多少能在检索上下文中找到依据。
   - 1.0：所有关键信息点都能找到
   - 0.8：大部分能找到，少量缺失
   - 0.5：约一半能找到
   - 0.2：仅少量能找到
   - 0.0：完全找不到

请以JSON格式返回四个分数，例如：
{{"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7, "context_recall": 0.6}}
只返回JSON，不要其他内容。"""

    try:
        from utils.llm_utils import get_llm_client
        client = get_llm_client(model=JUDGE_MODEL, json_mode=True)
        response = client.invoke([{"role": "user", "content": prompt}])
        result_text = response.content.strip()
        result = json.loads(result_text)
        return {
            "faithfulness": float(min(max(result.get("faithfulness", 0.0), 0.0), 1.0)),
            "answer_relevancy": float(min(max(result.get("answer_relevancy", 0.0), 0.0), 1.0)),
            "context_precision": float(min(max(result.get("context_precision", 0.0), 0.0), 1.0)),
            "context_recall": float(min(max(result.get("context_recall", 0.0), 0.0), 1.0)),
        }
    except Exception as e:
        print(f"    生成指标 LLM 评估失败: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}


# ====================================================================
# 7. 汇总统计
# ====================================================================
def calc_summary(results: List[Dict], top_k: int = 5) -> Dict[str, Any]:
    """计算整体汇总、分组统计、文档级 vs 内容级对比"""
    if not results:
        return {}

    def avg(key: str, subkey: Optional[str] = None) -> float:
        vals = []
        for r in results:
            if subkey:
                d = r.get(key)
                if d is None:
                    continue
                v = d.get(subkey)
            else:
                v = r.get(key)
            if v is not None:
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "config": {"top_k": top_k},
        "total": len(results),
        "doc_level_retrieval": {
            "avg_recall_at_k": avg("doc_metrics", "recall_at_k"),
            "avg_precision_at_k": avg("doc_metrics", "precision_at_k"),
            "avg_f1": avg("doc_metrics", "f1"),
            "avg_mrr": avg("doc_metrics", "mrr"),
            "recall_100_pct": round(sum(1 for r in results if r["doc_metrics"]["recall_at_k"] == 1.0) / len(results) * 100, 1),
            "recall_0_pct": round(sum(1 for r in results if r["doc_metrics"]["recall_at_k"] == 0.0) / len(results) * 100, 1),
        },
    }

    # 内容级（仅当有内容级结果时）
    has_content = any(r.get("content_metrics") is not None for r in results)
    if has_content:
        summary["content_level_retrieval"] = {
            "avg_precision_at_k": avg("content_metrics", "precision_at_k"),
            "avg_recall_approx": avg("content_metrics", "recall_approx"),
            "avg_f1": avg("content_metrics", "f1"),
            "avg_mrr": avg("content_metrics", "mrr"),
            "avg_hit_rate_at_k": avg("content_metrics", "hit_rate_at_k"),
        }
        # 文档级 vs 内容级对比
        d = summary["doc_level_retrieval"]
        c = summary["content_level_retrieval"]
        summary["doc_vs_content_comparison"] = {
            "precision_diff": round(c["avg_precision_at_k"] - d["avg_precision_at_k"], 4),
            "recall_diff": round(c["avg_recall_approx"] - d["avg_recall_at_k"], 4),
            "f1_diff": round(c["avg_f1"] - d["avg_f1"], 4),
            "mrr_diff": round(c["avg_mrr"] - d["avg_mrr"], 4),
            "interpretation": (
                "内容级 Precision 低于文档级，说明 item_name 命中了但 chunk 内容不一定有用；"
                "内容级 Recall 低于文档级，说明相关商品被检索到了但关键内容 chunk 未召回。"
            ),
        }

    # 生成指标
    has_gen = any(r.get("gen_metrics") and r["gen_metrics"].get("faithfulness", 0) > 0 for r in results)
    if has_gen:
        summary["generation"] = {
            "avg_faithfulness": avg("gen_metrics", "faithfulness"),
            "avg_answer_relevancy": avg("gen_metrics", "answer_relevancy"),
            "avg_context_precision": avg("gen_metrics", "context_precision"),
            "avg_context_recall": avg("gen_metrics", "context_recall"),
        }

    # 按问题类型分组
    by_type: Dict[str, List] = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "未知")].append(r)
    summary["by_question_type"] = {}
    for qtype, items in by_type.items():
        summary["by_question_type"][qtype] = _group_stats(items, has_content)

    # 按难度分组
    by_diff: Dict[str, List] = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty", "未知")].append(r)
    summary["by_difficulty"] = {}
    for diff, items in by_diff.items():
        summary["by_difficulty"][diff] = _group_stats(items, has_content)

    # 混淆测试专项
    confusion_ids = {f"TT-{i:03d}" for i in range(14, 24)}
    confusion_results = [r for r in results if r.get("sample_id") in confusion_ids]
    if confusion_results:
        summary["confusion_test"] = _group_stats(confusion_results, has_content)
        summary["confusion_test"]["recall_0_count"] = sum(
            1 for r in confusion_results if r["doc_metrics"]["recall_at_k"] == 0.0
        )

    return summary


def _group_stats(items: List[Dict], has_content: bool) -> Dict[str, Any]:
    """单组统计辅助函数"""
    n = len(items)
    stats = {"count": n}
    stats["doc_recall"] = round(sum(r["doc_metrics"]["recall_at_k"] for r in items) / n, 4)
    stats["doc_precision"] = round(sum(r["doc_metrics"]["precision_at_k"] for r in items) / n, 4)
    stats["doc_f1"] = round(sum(r["doc_metrics"]["f1"] for r in items) / n, 4)
    stats["doc_mrr"] = round(sum(r["doc_metrics"]["mrr"] for r in items) / n, 4)
    if has_content:
        cm = [r.get("content_metrics") or {} for r in items]
        stats["content_precision"] = round(sum(c.get("precision_at_k", 0) for c in cm) / n, 4)
        stats["content_recall"] = round(sum(c.get("recall_approx", 0) for c in cm) / n, 4)
        stats["content_f1"] = round(sum(c.get("f1", 0) for c in cm) / n, 4)
        stats["content_mrr"] = round(sum(c.get("mrr", 0) for c in cm) / n, 4)
    return stats


# ====================================================================
# 8. 报告打印
# ====================================================================
def print_report(summary: Dict[str, Any], total: int):
    """打印控制台汇总报告"""
    print("\n" + "=" * 72)
    print(f"  ThinkTank RAG 评估报告（共 {total} 条样本）")
    print("=" * 72)

    top_k = summary.get("../config", {}).get("top_k", 5)

    # 文档级检索指标
    print(f"\n【文档级检索指标】（item_name 匹配，Top-{top_k}）")
    d = summary["doc_level_retrieval"]
    print(f"  Recall@{top_k}:    {d['avg_recall_at_k']}  (完全召回: {d['recall_100_pct']}%, 零召回: {d['recall_0_pct']}%)")
    print(f"  Precision@{top_k}: {d['avg_precision_at_k']}")
    print(f"  F1:                {d['avg_f1']}")
    print(f"  MRR:               {d['avg_mrr']}")

    # 内容级检索指标
    if "content_level_retrieval" in summary:
        print(f"\n【内容级检索指标】（LLM 判断 chunk 相关性，Top-{top_k}）")
        c = summary["content_level_retrieval"]
        print(f"  Precision@{top_k}: {c['avg_precision_at_k']}")
        print(f"  Recall(近似):      {c['avg_recall_approx']}  (分母=相关商品总chunk数)")
        print(f"  F1:                {c['avg_f1']}")
        print(f"  MRR:               {c['avg_mrr']}")
        print(f"  Hit Rate@{top_k}:  {c['avg_hit_rate_at_k']}")

    # 对比
    if "doc_vs_content_comparison" in summary:
        cmp = summary["doc_vs_content_comparison"]
        print(f"\n【文档级 vs 内容级 对比】")
        print(f"  Precision 差异: {cmp['precision_diff']:+.4f}  (内容级 - 文档级)")
        print(f"  Recall    差异: {cmp['recall_diff']:+.4f}")
        print(f"  F1        差异: {cmp['f1_diff']:+.4f}")
        print(f"  MRR       差异: {cmp['mrr_diff']:+.4f}")
        print(f"  解读: {cmp['interpretation']}")

    # 生成指标
    if "generation" in summary:
        print(f"\n【生成指标】（LLM-as-Judge）")
        g = summary["generation"]
        print(f"  Faithfulness:      {g['avg_faithfulness']}")
        print(f"  Answer Relevancy:  {g['avg_answer_relevancy']}")
        print(f"  Context Precision: {g['avg_context_precision']}")
        print(f"  Context Recall:    {g['avg_context_recall']}")

    # 按问题类型
    if "by_question_type" in summary:
        print(f"\n【按问题类型分组】")
        for qtype, stats in summary["by_question_type"].items():
            line = f"  {qtype:10s} (n={stats['count']:2d}): 文档 Recall={stats['doc_recall']:.4f}, P={stats['doc_precision']:.4f}, F1={stats['doc_f1']:.4f}, MRR={stats['doc_mrr']:.4f}"
            if "content_precision" in stats:
                line += f" | 内容 P={stats['content_precision']:.4f}, R={stats['content_recall']:.4f}"
            print(line)

    # 按难度
    if "by_difficulty" in summary:
        print(f"\n【按难度分组】")
        for diff, stats in summary["by_difficulty"].items():
            line = f"  {diff:6s} (n={stats['count']:2d}): 文档 Recall={stats['doc_recall']:.4f}, F1={stats['doc_f1']:.4f}"
            if "content_precision" in stats:
                line += f" | 内容 P={stats['content_precision']:.4f}, R={stats['content_recall']:.4f}"
            print(line)

    # 混淆测试
    if "confusion_test" in summary:
        c = summary["confusion_test"]
        print(f"\n【混淆测试专项】(TT-014~TT-023, n={c['count']})")
        print(f"  文档级: Recall={c['doc_recall']}, Precision={c['doc_precision']}, F1={c['doc_f1']}, MRR={c['doc_mrr']}")
        if "content_precision" in c:
            print(f"  内容级: Precision={c['content_precision']}, Recall={c['content_recall']}, F1={c['content_f1']}")
        print(f"  零召回数: {c.get('recall_0_count', 'N/A')}/{c['count']}")

    print("\n" + "=" * 72)


# ====================================================================
# 9. 主流程
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="ThinkTank RAG 评估脚本 v2")
    parser.add_argument("--limit", type=int, default=None, help="只评估前N条")
    parser.add_argument("--confusion-only", action="store_true", help="只评估混淆测试10条")
    parser.add_argument("--skip-gen", action="store_true", help="跳过生成指标")
    parser.add_argument("--skip-content-judge", action="store_true", help="跳过内容级 LLM 判断")
    parser.add_argument("--top-k", type=int, default=5, help="检索 Top-K，默认5")
    parser.add_argument("--from-json", type=str, default=None,
                        help="从本地 JSON 文件加载评估样本（lark-cli 不可用时自动降级到 eval_samples.json）")
    args = parser.parse_args()

    print("=" * 72)
    print("ThinkTank RAG 系统评估（v2 - item_name 匹配 + 内容级判断）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: Top-K={args.top_k}, skip_gen={args.skip_gen}, "
          f"skip_content_judge={args.skip_content_judge}, confusion_only={args.confusion_only}")
    print("=" * 72)

    # 1. 加载评估样本
    samples = load_eval_samples(
        limit=args.limit,
        confusion_only=args.confusion_only,
        from_json=args.from_json,
    )

    # 2. 全量加载 Milvus 映射（只查一次）
    file_to_items, item_chunk_count, chunk_to_item, total_chunks = load_milvus_mappings()

    # 3. 逐条评估
    print(f"\n[3/6] 开始逐条评估（共 {len(samples)} 条）...")
    results = []

    for idx, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        question = sample["question"]
        print(f"\n  [{idx}/{len(samples)}] {sid}: {question[:60]}...")

        # 3.1 标注转换：file_title -> item_name 集合
        relevant_item_names = convert_relevant_to_item_names(
            sample["relevant_doc_ids"], file_to_items, sample_id=sid
        )
        print(f"    标注相关 item_name: {sorted(relevant_item_names)}")

        # 3.2 调用 Query 流程
        session_id = f"eval_{sid}_{int(time.time())}"
        query_result = run_query_workflow(question, session_id)

        if query_result["error"]:
            print(f"    Query流程异常: {str(query_result['error'])[:120]}")

        # 3.3 给 reranked_docs 补 item_name
        reranked_docs = enrich_reranked_docs_with_item_name(
            query_result["reranked_docs"], chunk_to_item
        )
        contexts = [doc.get("content", "") for doc in reranked_docs]
        answer = query_result["answer"]

        local_count = sum(1 for d in reranked_docs if d.get("source") != "web")
        web_count = len(reranked_docs) - local_count
        print(f"    检索到 {len(reranked_docs)} 条 (local={local_count}, web={web_count}), 答案长度: {len(answer)}")

        # 3.4 文档级检索指标（item_name 匹配）
        doc_metrics = calc_doc_level_metrics(reranked_docs, relevant_item_names, k=args.top_k)
        print(f"    [文档级] Recall={doc_metrics['recall_at_k']}, "
              f"Precision={doc_metrics['precision_at_k']}, "
              f"F1={doc_metrics['f1']}, MRR={doc_metrics['mrr']}")

        # 3.5 内容级相关性判断（LLM）
        content_metrics = None
        if not args.skip_content_judge and reranked_docs:
            print(f"    正在判断内容级相关性（LLM, {len(reranked_docs)} 个 chunk）...")
            judgments = judge_chunks_relevance_batch(question, reranked_docs)
            content_metrics = calc_content_level_metrics(
                reranked_docs, judgments, relevant_item_names, item_chunk_count, k=args.top_k
            )
            print(f"    [内容级] Precision={content_metrics['precision_at_k']}, "
                  f"Recall(近似)={content_metrics['recall_approx']}, "
                  f"F1={content_metrics['f1']}, MRR={content_metrics['mrr']}, "
                  f"HitRate={content_metrics['hit_rate_at_k']}")

        # 3.6 生成指标（LLM-as-Judge）
        gen_metrics = {
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
        }
        if not args.skip_gen and answer:
            print(f"    正在计算生成指标（LLM-as-Judge）...")
            gen_metrics = calc_generation_metrics(
                question, answer, sample["ground_truth"], contexts, relevant_item_names
            )
            print(f"    [生成] Faith={gen_metrics['faithfulness']}, "
                  f"Rel={gen_metrics['answer_relevancy']}, "
                  f"CP={gen_metrics['context_precision']}, "
                  f"CR={gen_metrics['context_recall']}")

        # 3.7 保存单条结果
        results.append({
            "sample_id": sid,
            "category": sample["category"],
            "question_type": sample["question_type"],
            "difficulty": sample["difficulty"],
            "question": question,
            "ground_truth": sample["ground_truth"],
            "relevant_doc_ids": sample["relevant_doc_ids"],
            "relevant_item_names": sorted(relevant_item_names),
            "retrieved_count": len(reranked_docs),
            "retrieved_local_count": local_count,
            "retrieved_web_count": web_count,
            "retrieved_item_names": doc_metrics.get("retrieved_item_names", []),
            "answer": answer,
            "answer_length": len(answer),
            "item_names_recognized": query_result["item_names"],
            "rewritten_query": query_result["rewritten_query"],
            "error": query_result["error"],
            "doc_metrics": doc_metrics,
            "content_metrics": content_metrics,
            "gen_metrics": gen_metrics,
        })

        # 避免请求过快
        if not args.skip_gen or not args.skip_content_judge:
            time.sleep(0.5)

    # 4. 汇总统计
    print(f"\n[4/6] 汇总统计...")
    summary = calc_summary(results, top_k=args.top_k)

    # 5. 保存结果
    print(f"\n[5/6] 保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"eval_result_{timestamp}.json")

    output_data = {
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": "v2",
        "config": {
            "top_k": args.top_k,
            "skip_gen": args.skip_gen,
            "skip_content_judge": args.skip_content_judge,
            "confusion_only": args.confusion_only,
            "total_samples": len(samples),
            "milvus_total_chunks": total_chunks,
        },
        "summary": summary,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {output_file}")

    # 6. 打印报告
    print(f"\n[6/6] 评估报告")
    print_report(summary, len(samples))
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
