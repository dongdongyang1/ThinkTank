"""
ThinkTank Agent 模式检索效果评估脚本
====================================
基于 eval_rag.py 的评估框架，调用 Agent 模式（KBQueryAgent）而非标准 RAG 流程。
评估混淆测试10条（TT-014~TT-023），计算三类指标：
  1. 文档级检索指标（item_name 匹配）
  2. 内容级检索指标（LLM 判断 chunk 相关性）
  3. 生成指标（LLM-as-Judge 评估答案质量）

用法：
  python eval_agent.py                          # 评估混淆10条，三个指标都算
  python eval_agent.py --skip-gen               # 跳过生成指标
  python eval_agent.py --skip-content-judge     # 跳过内容级 LLM 判断
  python eval_agent.py --limit 5                # 只测前5条
"""

import os
import sys
import re
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

# ==================== 路径配置 ====================
PROJECT_ROOT = r"D:\ThinkTank"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

# 复用 eval_rag.py 中的评估函数
from eval_rag import (
    load_eval_samples,
    load_milvus_mappings,
    convert_relevant_to_item_names,
    calc_doc_level_metrics,
    judge_chunks_relevance_batch,
    calc_content_level_metrics,
    calc_generation_metrics,
    calc_summary,
    print_report,
    OUTPUT_DIR,
)

from processor.agent_processor.agent_graph import KBQueryAgent
from langchain_core.messages import ToolMessage


# ====================================================================
# 从 Agent 的 ToolMessage 中解析检索结果
# ====================================================================
def parse_docs_from_tool_messages(messages: list) -> List[Dict[str, Any]]:
    """
    从 Agent 的所有 ToolMessage 中解析检索到的文档列表。
    kb_search 返回的格式：
      [文档1] 标题: xxx | 来源: local | 相关度: 0.89
      内容: ...
      ---
      [文档2] ...
    """
    all_docs = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content or ""
        docs = _parse_kb_search_output(content)
        all_docs.extend(docs)
    return all_docs


def _parse_kb_search_output(content: str) -> List[Dict[str, Any]]:
    """解析 kb_search 工具返回的文本格式为结构化文档列表"""
    docs = []
    # 按 [文档N] 分割
    parts = re.split(r'\[文档\d+\]', content)
    for part in parts[1:]:  # 跳过第一个（[文档1]之前的内容）
        part = part.strip()
        if not part:
            continue

        doc = {
            "content": "",
            "source": "local",
            "relevance_score": 0.0,
            "title": "",
            "chunks_id": None,
            "item_name": "",
        }

        # 解析第一行：标题: xxx | 来源: local | 商品: xxx | 相关度: 0.89
        first_line_end = part.find('\n')
        if first_line_end > 0:
            header = part[:first_line_end]
            body = part[first_line_end + 1:]
        else:
            header = part
            body = ""

        # 解析标题
        title_match = re.search(r'标题:\s*([^|]+)', header)
        if title_match:
            doc["title"] = title_match.group(1).strip()

        # 解析来源
        source_match = re.search(r'来源:\s*([^|]+)', header)
        if source_match:
            doc["source"] = source_match.group(1).strip()

        # 解析商品名
        item_match = re.search(r'商品:\s*([^|]+)', header)
        if item_match:
            doc["item_name"] = item_match.group(1).strip()

        # 解析相关度
        score_match = re.search(r'相关度:\s*([\d.]+)', header)
        if score_match:
            try:
                doc["relevance_score"] = float(score_match.group(1))
            except ValueError:
                pass

        # 解析内容（去掉"内容:"前缀，去掉末尾的 ---）
        body = re.sub(r'^内容:\s*', '', body)
        body = re.sub(r'\n---\s*$', '', body.strip())
        doc["content"] = body.strip()

        if doc["content"]:
            docs.append(doc)

    return docs


# ====================================================================
# 调用 Agent 模式
# ====================================================================
def run_agent_query(question: str, session_id: str) -> Dict[str, Any]:
    """
    调用 Agent 模式（KBQueryAgent）。
    返回：{messages, answer, reranked_docs, error}
    """
    agent = KBQueryAgent()
    init_state = {
        "messages": [("user", question)],
        "session_id": session_id,
        "is_stream": False,
        "query_intent": "",
        "rewritten_query": "",
        "retrieved_docs": [],
        "loop_count": 0,
        "reflection_count": 0,
        "long_term_memory": "",
    }

    try:
        final_state = agent.run(init_state, stream=False)
        messages = final_state.get("messages", [])
        answer = messages[-1].content if messages else ""

        # 从 ToolMessage 解析检索结果
        reranked_docs = parse_docs_from_tool_messages(messages)

        return {
            "messages": messages,
            "answer": answer,
            "reranked_docs": reranked_docs,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "messages": [],
            "answer": "",
            "reranked_docs": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ====================================================================
# 主流程
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="ThinkTank Agent 模式评估")
    parser.add_argument("--limit", type=int, default=None, help="只评估前N条")
    parser.add_argument("--confusion-only", action="store_true", default=True,
                        help="只评估混淆测试10条（默认开启）")
    parser.add_argument("--skip-gen", action="store_true", help="跳过生成指标")
    parser.add_argument("--skip-content-judge", action="store_true", help="跳过内容级 LLM 判断")
    parser.add_argument("--top-k", type=int, default=5, help="检索 Top-K，默认5")
    parser.add_argument("--from-json", type=str, default=None, help="从本地 JSON 加载评估样本")
    args = parser.parse_args()

    print("=" * 72)
    print("ThinkTank Agent 模式检索效果评估")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: Top-K={args.top_k}, skip_gen={args.skip_gen}, "
          f"skip_content_judge={args.skip_content_judge}, confusion_only=True")
    print("=" * 72)

    # 1. 加载评估样本（混淆10条）
    samples = load_eval_samples(
        limit=args.limit,
        confusion_only=True,
        from_json=args.from_json,
    )

    # 2. 全量加载 Milvus 映射
    file_to_items, item_chunk_count, chunk_to_item, total_chunks = load_milvus_mappings()

    # 3. 逐条评估
    print(f"\n[3/6] 开始逐条评估 Agent 模式（共 {len(samples)} 条）...")
    results = []

    for idx, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        question = sample["question"]
        print(f"\n  [{idx}/{len(samples)}] {sid}: {question[:60]}...")

        # 3.1 标注转换
        relevant_item_names = convert_relevant_to_item_names(
            sample["relevant_doc_ids"], file_to_items, sample_id=sid
        )
        print(f"    标注相关 item_name: {sorted(relevant_item_names)}")

        # 3.2 调用 Agent 模式
        session_id = f"eval_agent_{sid}_{int(time.time())}"
        t0 = time.time()
        query_result = run_agent_query(question, session_id)
        elapsed = time.time() - t0

        if query_result["error"]:
            print(f"    Agent 异常: {str(query_result['error'])[:120]}")

        reranked_docs = query_result["reranked_docs"]
        answer = query_result["answer"]
        contexts = [doc.get("content", "") for doc in reranked_docs]

        local_count = sum(1 for d in reranked_docs if d.get("source") != "web")
        web_count = len(reranked_docs) - local_count
        print(f"    检索到 {len(reranked_docs)} 条 (local={local_count}, web={web_count}), "
              f"答案长度: {len(answer)}, 耗时: {elapsed:.1f}s")

        # 3.3 文档级检索指标
        doc_metrics = calc_doc_level_metrics(reranked_docs, relevant_item_names, k=args.top_k)
        print(f"    [文档级] Recall={doc_metrics['recall_at_k']}, "
              f"Precision={doc_metrics['precision_at_k']}, "
              f"F1={doc_metrics['f1']}, MRR={doc_metrics['mrr']}")

        # 3.4 内容级相关性判断
        content_metrics = None
        if not args.skip_content_judge and reranked_docs:
            print(f"    正在判断内容级相关性（LLM, {len(reranked_docs)} 个 chunk）...")
            judgments = judge_chunks_relevance_batch(question, reranked_docs)
            content_metrics = calc_content_level_metrics(
                reranked_docs, judgments, relevant_item_names, item_chunk_count, k=args.top_k
            )
            print(f"    [内容级] Precision={content_metrics['precision_at_k']}, "
                  f"Recall(近似)={content_metrics['recall_approx']}, "
                  f"F1={content_metrics['f1']}, MRR={content_metrics['mrr']}")

        # 3.5 生成指标
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

        # 3.6 保存单条结果
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
            "elapsed_seconds": round(elapsed, 2),
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
    output_file = os.path.join(OUTPUT_DIR, f"eval_agent_result_{timestamp}.json")

    output_data = {
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": "agent_v1",
        "config": {
            "top_k": args.top_k,
            "skip_gen": args.skip_gen,
            "skip_content_judge": args.skip_content_judge,
            "confusion_only": True,
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
    print(f"\n[6/6] 评估报告（Agent 模式）")
    print_report(summary, len(samples))
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
