"""
RAGAS 评估脚本
- 加载测试集，逐条调用 query API
- 组装 RAGAS 数据集，计算全部指标
- 按类型分组统计，生成 JSON + HTML 报告
"""
import os
import sys
import json
import time
import requests
from typing import List, Dict
from datetime import datetime

# 修复 numpy 在 Python 3.12 + numpy 1.26 上移除的类型别名（transformers/FlagEmbedding 依赖）
import numpy
for _attr, _default in [
    ('long', int), ('ulong', int), ('int', int),
    ('float', float), ('complex', complex), ('bool', bool),
    ('object', object), ('str', str), ('unicode', str),
    ('longlong', int), ('ulonglong', int),
    ('int8', int), ('int16', int), ('int32', int), ('int64', int),
    ('uint8', int), ('uint16', int), ('uint32', int), ('uint64', int),
    ('float16', float), ('float32', float), ('float64', float),
]:
    if not hasattr(numpy, _attr):
        setattr(numpy, _attr, _default)

# 强制 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = r"D:\ThinkTank"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

# ========== 配置 ==========
API_BASE = "http://localhost:8002"
API_KEY = os.getenv("API_KEY", "abcd1234")
DATASET_FILE = "eval/eval_samples_v2.json"
OUTPUT_DIR = "eval/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RAGAS Judge 模型配置（用 qwen-plus，便宜够用）
JUDGE_MODEL = "qwen-plus"
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# BGE-M3 本地模型配置（用于 embedding，避免 DashScope 格式兼容问题）
BGE_M3_PATH = r"D:\modelscope_cache\models\BAAI--bge-m3\snapshots\master"
BGE_DEVICE = "cuda:0"
BGE_USE_FP16 = True

# 测试样本数（None = 全部）
SAMPLE_LIMIT = None
# 每条请求间隔（秒），避免 API 限流
REQUEST_INTERVAL = 1


def load_dataset() -> List[Dict]:
    """加载测试集"""
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if SAMPLE_LIMIT:
        data = data[:SAMPLE_LIMIT]
    print(f"加载测试集: {len(data)} 条")
    return data


def query_api(question: str, session_id: str = "eval_session") -> Dict:
    """调用 query API（非流式），返回 answer 和 retrieved_contexts"""
    url = f"{API_BASE}/query"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }
    payload = {
        "query": question,
        "session_id": session_id,
        "user_id": "eval_user",
        "is_stream": False,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    API 调用失败: {e}")
        return {"answer": "", "retrieved_contexts": [], "error": str(e)}


def build_ragas_dataset(samples: List[Dict]) -> tuple:
    """逐条调用 API，组装 RAGAS 数据集（带缓存，避免重复调用 API）"""
    from datasets import Dataset

    # 缓存文件：API 调用结果持久化，重新跑评估不需要重跑 API
    cache_file = os.path.join(OUTPUT_DIR, "api_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"加载 API 缓存: {len(cache)} 条已缓存")

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    metadata_list = []

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        question = sample["question"]
        gt = sample["ground_truth"]
        qtype = sample["question_type"]

        print(f"[{i+1}/{len(samples)}] {sid} [{qtype}]: {question[:40]}...")

        # 优先用缓存，缓存没有才调用 API
        if sid in cache and cache[sid].get("answer"):
            result = cache[sid]
            print(f"    [缓存命中] 答案长度: {len(result.get('answer', ''))}, 检索文档: {len(result.get('retrieved_contexts', []))} 条")
        else:
            result = query_api(question, session_id=f"eval_{sid}")
            cache[sid] = result
            # 每调完一条就存一次缓存，防止中途失败丢失
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"    [API调用] 答案长度: {len(result.get('answer', ''))}, 检索文档: {len(result.get('retrieved_contexts', []))} 条")

        answer = result.get("answer", "")
        retrieved = result.get("retrieved_contexts", [])

        # 提取文档内容列表
        ctx_list = [str(c.get("content", "")).strip() for c in retrieved if c.get("content")]
        ctx_list = [c for c in ctx_list if c]  # 过滤空字符串
        if not ctx_list:
            ctx_list = ["无相关文档"]  # RAGAS 要求 contexts 非空，且必须是字符串

        # 数据清洗：确保所有字段都是非空字符串（防止embedding报参数错误）
        clean_question = str(question).strip() if question else "空问题"
        clean_answer = str(answer).strip() if answer else "无答案"
        clean_gt = str(gt).strip() if gt else "无参考答案"

        questions.append(clean_question)
        answers.append(clean_answer)
        contexts.append(ctx_list)
        ground_truths.append(clean_gt)
        metadata_list.append({
            "sample_id": sid,
            "question_type": qtype,
            "category": sample.get("category", ""),
            "difficulty": sample.get("difficulty", ""),
        })

        time.sleep(REQUEST_INTERVAL)

    # 构建 RAGAS Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    return dataset, metadata_list


def run_ragas_evaluation(dataset):
    """运行 RAGAS 评估，计算全部指标"""
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    # 配置 Judge LLM（qwen-max）
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # 配置 Embeddings（用 BGE-M3 本地模型，直接用 transformers 实现，避免 FlagEmbedding 版本兼容问题）
    print(f"加载 BGE-M3 本地模型: {BGE_M3_PATH}")
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings
    import torch
    from transformers import AutoTokenizer, AutoModel

    class BGEM3LocalEmbeddings(Embeddings):
        """BGE-M3 本地模型的 LangChain Embeddings 包装（直接用 transformers 实现 dense 向量）"""
        def __init__(self, model_path, device, use_fp16):
            self._device = device
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self._model.to(device)
            if use_fp16:
                self._model.half()
            self._model.eval()

        def _encode(self, texts):
            encoded = self._tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 的输出作为 dense 向量
            last_hidden = outputs.last_hidden_state[:, 0, :]
            # L2 归一化
            norms = torch.norm(last_hidden, p=2, dim=1, keepdim=True)
            normalized = last_hidden / norms
            return normalized.cpu().float().numpy()

        def embed_documents(self, texts):
            # 分批处理，避免显存溢出
            batch_size = 8
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                all_embeddings.append(self._encode(batch))
            import numpy as np
            return np.concatenate(all_embeddings, axis=0).tolist()

        def embed_query(self, text):
            return self._encode([text])[0].tolist()

    embeddings = BGEM3LocalEmbeddings(BGE_M3_PATH, BGE_DEVICE, BGE_USE_FP16)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    print("BGE-M3 本地模型加载完成")

    # 指标列表（4个核心指标，全保留）
    metrics_list = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    print(f"\n开始 RAGAS 评估，指标: {[m.name for m in metrics_list]}")
    print(f"Judge 模型: {JUDGE_MODEL}")
    print(f"并发数: 5（RunConfig max_workers=5）")
    print(f"API 缓存: 已启用（不重跑API，只跑评估）")

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=5)

    result = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )

    return result


def analyze_results(result, metadata_list: List[Dict], samples: List[Dict]):
    """分析评估结果，按类型分组统计"""
    import pandas as pd

    df = result.to_pandas()

    # 添加元数据
    df["sample_id"] = [m["sample_id"] for m in metadata_list]
    df["question_type"] = [m["question_type"] for m in metadata_list]
    df["category"] = [m["category"] for m in metadata_list]
    df["difficulty"] = [m["difficulty"] for m in metadata_list]

    # 总体指标（只选数值类型的列，排除数据集原始列）
    exclude_cols = [
        "question", "answer", "contexts", "ground_truth",
        "sample_id", "question_type", "category", "difficulty",
        "user_input", "retrieved_contexts", "response", "reference",
    ]
    metric_cols = [c for c in df.columns if c not in exclude_cols]
    # 把指标列转成数值类型（RAGAS 0.4.3 某些列可能是字符串）
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 只保留有有效值的指标列
    metric_cols = [c for c in metric_cols if df[c].notna().any()]
    overall = df[metric_cols].mean().to_dict()

    # 按类型分组
    by_type = df.groupby("question_type")[metric_cols].mean().to_dict("index")

    # 按难度分组
    by_difficulty = df.groupby("difficulty")[metric_cols].mean().to_dict("index")

    # 按检索来源分组（区分纯RAG vs 长期记忆 vs 拒答类问题）
    # 兼容不同版本的列名（contexts 或 retrieved_contexts）
    ctx_col = "contexts" if "contexts" in df.columns else "retrieved_contexts"
    df["has_retrieved_context"] = df[ctx_col].apply(lambda x: len(x) > 0 and any(c.strip() for c in x))
    # 拒答类问题（知识库无答案）
    df["is_refusal"] = df["question_type"].str.contains("拒答|无答案", na=False)
    # 分组：拒答类 / RAG检索 / 长期记忆
    df["retrieval_group"] = df.apply(
        lambda row: "拒答类问题" if row["is_refusal"]
        else ("RAG检索回答" if row["has_retrieved_context"] else "长期记忆回答"),
        axis=1,
    )
    by_retrieval = df.groupby("retrieval_group")[metric_cols].mean().to_dict("index")
    retrieval_count = df["retrieval_group"].value_counts().to_dict()

    return df, overall, by_type, by_difficulty, by_retrieval, retrieval_count


def save_results(df, overall, by_type, by_difficulty, by_retrieval, retrieval_count):
    """保存评估结果（JSON + CSV + HTML）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. JSON 报告
    report = {
        "timestamp": timestamp,
        "total_samples": len(df),
        "judge_model": JUDGE_MODEL,
        "overall_metrics": {k: round(v, 4) for k, v in overall.items()},
        "by_type": {k: {m: round(v, 4) for m, v in metrics.items()} for k, metrics in by_type.items()},
        "by_difficulty": {k: {m: round(v, 4) for m, v in metrics.items()} for k, metrics in by_difficulty.items()},
        "by_retrieval": {k: {m: round(v, 4) for m, v in metrics.items()} for k, metrics in by_retrieval.items()},
        "retrieval_count": retrieval_count,
    }
    json_path = os.path.join(OUTPUT_DIR, f"ragas_report_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告: {json_path}")

    # 2. CSV 明细
    csv_path = os.path.join(OUTPUT_DIR, f"ragas_details_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 明细: {csv_path}")

    # 3. HTML 报告
    html_path = os.path.join(OUTPUT_DIR, f"ragas_report_{timestamp}.html")
    _generate_html_report(report, html_path)
    print(f"HTML 报告: {html_path}")

    return report


def _generate_html_report(report: Dict, html_path: str):
    """生成 HTML 评估报告"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS 评估报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-card .value {{ font-size: 28px; font-weight: bold; }}
        .metric-card .label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4a90d9; color: white; }}
        tr:hover {{ background: #f0f7ff; }}
        .score-high {{ color: #2e7d32; font-weight: bold; }}
        .score-mid {{ color: #f57c00; font-weight: bold; }}
        .score-low {{ color: #c62828; font-weight: bold; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RAGAS 评估报告</h1>
        <div class="meta">
            <p>评估时间: {report['timestamp']} | 样本数: {report['total_samples']} | Judge 模型: {report['judge_model']}</p>
        </div>

        <div class="summary">
            <h2>总体指标</h2>
            <div class="metric-grid">
"""

    for metric, value in report["overall_metrics"].items():
        pct = value * 100
        color_class = "score-high" if pct >= 70 else ("score-mid" if pct >= 40 else "score-low")
        html += f"""
                <div class="metric-card">
                    <div class="value">{pct:.1f}%</div>
                    <div class="label">{metric}</div>
                </div>"""

    html += """
            </div>
        </div>

        <div class="summary" style="margin-top: 20px;">
            <h2>按问题类型分组</h2>
            <table>
                <tr><th>问题类型</th>"""

    # 表头
    first_type = list(report["by_type"].keys())[0]
    for metric in report["by_type"][first_type].keys():
        html += f"<th>{metric}</th>"
    html += "</tr>"

    # 数据行
    for qtype, metrics in report["by_type"].items():
        html += f"<tr><td><b>{qtype}</b></td>"
        for value in metrics.values():
            pct = value * 100
            color_class = "score-high" if pct >= 70 else ("score-mid" if pct >= 40 else "score-low")
            html += f'<td class="{color_class}">{pct:.1f}%</td>'
        html += "</tr>"

    html += """
            </table>
        </div>

        <div class="summary" style="margin-top: 20px;">
            <h2>按难度分组</h2>
            <table>
                <tr><th>难度</th>"""

    first_diff = list(report["by_difficulty"].keys())[0]
    for metric in report["by_difficulty"][first_diff].keys():
        html += f"<th>{metric}</th>"
    html += "</tr>"

    for diff, metrics in report["by_difficulty"].items():
        html += f"<tr><td><b>{diff}</b></td>"
        for value in metrics.values():
            pct = value * 100
            color_class = "score-high" if pct >= 70 else ("score-mid" if pct >= 40 else "score-low")
            html += f'<td class="{color_class}">{pct:.1f}%</td>'
        html += "</tr>"

    html += """
            </table>
        </div>

        <div class="summary" style="margin-top: 20px;">
            <h2>按检索来源分组（RAG vs 长期记忆）</h2>
            <table>
                <tr><th>检索来源</th><th>样本数</th>"""

    # 表头
    first_ret = list(report["by_retrieval"].keys())[0]
    for metric in report["by_retrieval"][first_ret].keys():
        html += f"<th>{metric}</th>"
    html += "</tr>"

    # 数据行
    for ret_type, metrics in report["by_retrieval"].items():
        count = report["retrieval_count"].get(ret_type, 0)
        html += f"<tr><td><b>{ret_type}</b></td><td>{count}</td>"
        for value in metrics.values():
            pct = value * 100
            color_class = "score-high" if pct >= 70 else ("score-mid" if pct >= 40 else "score-low")
            html += f'<td class="{color_class}">{pct:.1f}%</td>'
        html += "</tr>"

    html += """
            </table>
        </div>
    </div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 60)
    print("RAGAS 评估脚本")
    print("=" * 60)

    # 1. 加载测试集
    samples = load_dataset()

    # 2. 逐条调用 API，组装 RAGAS 数据集
    print(f"\n{'=' * 60}")
    print("步骤 1/3: 调用 API 获取答案和检索结果")
    print(f"{'=' * 60}")
    dataset, metadata_list = build_ragas_dataset(samples)

    # 3. 运行 RAGAS 评估
    print(f"\n{'=' * 60}")
    print("步骤 2/3: 运行 RAGAS 评估")
    print(f"{'=' * 60}")
    result = run_ragas_evaluation(dataset)

    # 4. 分析结果
    print(f"\n{'=' * 60}")
    print("步骤 3/3: 分析结果并生成报告")
    print(f"{'=' * 60}")
    df, overall, by_type, by_difficulty, by_retrieval, retrieval_count = analyze_results(result, metadata_list, samples)

    # 5. 保存结果
    report = save_results(df, overall, by_type, by_difficulty, by_retrieval, retrieval_count)

    # 6. 打印总体指标
    print(f"\n{'=' * 60}")
    print("评估完成! 总体指标:")
    print(f"{'=' * 60}")
    for metric, value in report["overall_metrics"].items():
        print(f"  {metric:30s}: {value*100:.2f}%")
    print(f"\n报告已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
