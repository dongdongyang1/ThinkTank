"""
ThinkTank 测试集生成脚本
========================
从 Milvus 拉取商品名和文档内容，用 qwen-max 按 12 种类型批量生成测试集。

用法：
  python generate_eval_dataset.py
  python generate_eval_dataset.py --limit 50  # 只生成50条
  python generate_eval_dataset.py --dry-run    # 只拉取数据，不调用LLM
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
from collections import defaultdict

# 强制 UTF-8 编码，解决 Windows 中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==================== 路径配置 ====================
PROJECT_ROOT = r"D:\ThinkTank"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient, Collection
from openai import OpenAI

# ==================== 配置 ====================
MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")
CHUNKS_COLLECTION = os.getenv("CHUNKS_COLLECTION", "kb_chunks")
ITEM_NAME_COLLECTION = os.getenv("ITEM_NAME_COLLECTION", "kb_item_names")

# 生成测试集用的模型（必须比项目用的 qwen-plus 强）
GEN_MODEL = "qwen-max"
GEN_TEMPERATURE = 0.7  # 生成测试集需要一定的创造性

# 输出文件
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "eval", "eval_samples_v2.json")

# ==================== 测试集类型配置 ====================
# 按用户确认的表格
DATASET_TYPES = [
    {"type": "简单事实查询", "count": 15, "desc": "单文档单段落可直接回答", "difficulty": "简单"},
    {"type": "操作步骤查询", "count": 10, "desc": "多步骤操作流程", "difficulty": "中等"},
    {"type": "多跳推理", "count": 10, "desc": "需跨文档/跨段落关联推理", "difficulty": "困难"},
    {"type": "总结归纳", "count": 8, "desc": "需从多个片段归纳总结", "difficulty": "中等"},
    {"type": "对比查询", "count": 5, "desc": "多个产品/型号对比", "difficulty": "中等"},
    {"type": "模糊查询", "count": 5, "desc": "查询意图不明确，需消歧", "difficulty": "中等"},
    {"type": "拒答(知识库无答案)", "count": 10, "desc": "知识库中完全没有相关信息", "difficulty": "简单"},
    {"type": "无答案(部分相关但无直接答案)", "count": 8, "desc": "有相关文档但不包含具体答案", "difficulty": "中等"},
    {"type": "故障排查", "count": 8, "desc": "多步排查，需关联原因和解决方案", "difficulty": "困难"},
    {"type": "安全/边界问题", "count": 5, "desc": "涉及安全操作、警告信息", "difficulty": "中等"},
    {"type": "超长文本理解", "count": 3, "desc": "需理解长段落/复杂结构", "difficulty": "困难"},
    {"type": "同义词/口语化查询", "count": 5, "desc": "用口语化、同义词表达", "difficulty": "简单"},
]

TOTAL_COUNT = sum(t["count"] for t in DATASET_TYPES)


# ====================================================================
# 1. 从 Milvus 拉取数据
# ====================================================================
def fetch_knowledge_base() -> Dict[str, Any]:
    """从 Milvus 拉取商品名、文档列表、文档内容样本"""
    print(f"[1/4] 连接 Milvus: {MILVUS_URL}")
    client = MilvusClient(uri=MILVUS_URL)

    # 1. 获取所有商品名
    print(f"  拉取商品名集合: {ITEM_NAME_COLLECTION}")
    item_names = []
    try:
        results = client.query(
            collection_name=ITEM_NAME_COLLECTION,
            filter="",
            limit=100,
            output_fields=["item_name", "file_title", "category"]
        )
        for r in results:
            item_names.append({
                "item_name": r.get("item_name", ""),
                "file_title": r.get("file_title", ""),
                "category": r.get("category", ""),
            })
        print(f"  商品名数量: {len(item_names)}")
    except Exception as e:
        print(f"  拉取商品名失败: {e}")

    # 2. 获取所有文档标题（去重）
    print(f"  拉取文档标题集合: {CHUNKS_COLLECTION}")
    doc_titles = set()
    try:
        # 用分页查询获取所有 file_title
        offset = 0
        batch_size = 500
        while True:
            results = client.query(
                collection_name=CHUNKS_COLLECTION,
                filter="",
                limit=batch_size,
                offset=offset,
                output_fields=["file_title"]
            )
            if not results:
                break
            for r in results:
                title = r.get("file_title", "")
                if title:
                    doc_titles.add(title)
            offset += batch_size
            if offset > 10000:  # 安全上限
                break
        print(f"  文档标题数量: {len(doc_titles)}")
    except Exception as e:
        print(f"  拉取文档标题失败: {e}")

    # 3. 为每个文档抽样 2~3 条 chunk 内容
    print("  抽样文档内容...")
    doc_contents = defaultdict(list)
    try:
        for title in doc_titles:
            # 对每个文档查询前几条 chunk
            results = client.query(
                collection_name=CHUNKS_COLLECTION,
                filter=f'file_title == "{title}"',
                limit=3,
                output_fields=["content", "file_title"]
            )
            for r in results:
                content = r.get("content", "")
                if content and len(content) > 20:  # 过滤过短的chunk
                    doc_contents[title].append({
                        "content": content[:500],  # 截断，避免上下文过长
                    })
            time.sleep(0.05)  # 避免查询过快
        print(f"  有内容的文档数量: {len([t for t, c in doc_contents.items() if c])}")
    except Exception as e:
        print(f"  抽样文档内容失败: {e}")

    client.close()

    return {
        "item_names": item_names,
        "doc_titles": sorted(list(doc_titles)),
        "doc_contents": dict(doc_contents),
    }


# ====================================================================
# 2. 构建 LLM 提示词
# ====================================================================
def build_generation_prompt(
    dataset_type: Dict,
    kb_data: Dict,
    batch_size: int,
    start_id: int
) -> str:
    """构建测试集生成提示词"""

    # 准备知识库上下文（商品名 + 文档标题 + 部分文档内容）
    item_names_text = "\n".join([
        f"- {item['item_name']}（{item.get('category', '未分类')}）"
        for item in kb_data["item_names"][:30]
    ])

    doc_titles_text = "\n".join([f"- {t}" for t in kb_data["doc_titles"][:30]])

    # 抽样一些文档内容作为参考
    sample_docs = []
    for title, chunks in list(kb_data["doc_contents"].items())[:5]:
        if chunks:
            sample_text = "\n".join([c["content"][:200] for c in chunks[:2]])
            sample_docs.append(f"【{title}】\n{sample_text}")
    sample_docs_text = "\n\n".join(sample_docs)

    # 拒答类的特殊提示
    is_refusal = "拒答" in dataset_type["type"] or "无答案" in dataset_type["type"]

    prompt = f"""你是一位专业的RAG系统测试集设计专家。请基于以下知识库内容，生成 {batch_size} 条【{dataset_type['type']}】类型的测试问题。

## 知识库概览

### 商品/设备列表（共{len(kb_data['item_names'])}个）
{item_names_text}

### 文档标题列表（共{len(kb_data['doc_titles'])}个）
{doc_titles_text}

### 文档内容样本（仅供参考，实际知识库有更多内容）
{sample_docs_text}

## 生成要求

**类型**：{dataset_type['type']}
**说明**：{dataset_type['desc']}
**难度**：{dataset_type['difficulty']}

"""

    if is_refusal:
        prompt += """
## 拒答/无答案类特殊要求

这类问题的答案**不在知识库中**，系统应该拒答或说明无法找到答案。

- 拒答类：构造知识库中完全没有的产品/功能/问题（比如其他品牌的产品、知识库中没有的型号）
- 无答案类：构造与知识库主题相关但具体答案不在文档中的问题（比如问某个产品的价格、保修期，而文档里没写）

生成时请确保：
1. 问题看起来合理，不是明显的瞎编
2. ground_truth 写"知识库中无相关信息，应拒答"
3. relevant_doc_ids 留空（拒答类）或写最接近的文档（无答案类）
"""
    else:
        prompt += """
## 答案要求

1. ground_truth 必须基于知识库内容，准确、完整
2. relevant_doc_ids 填写能回答该问题的文档标题（1~3个）
3. 问题要自然、符合用户真实提问习惯，不要太书面化
4. 多跳推理类必须需要跨2个以上文档或段落才能回答
5. 操作步骤类必须包含完整的步骤序列
6. 故障排查类必须包含原因分析和排查步骤
"""

    prompt += f"""
## 输出格式

严格输出 JSON 数组，每条包含以下字段：
{{
  "sample_id": "TT-{start_id:03d}",
  "category": "问题所属产品/类别",
  "question": "用户问题",
  "question_type": "{dataset_type['type']}",
  "ground_truth": "参考答案（准确完整）",
  "relevant_doc_ids": ["相关文档标题1", "相关文档标题2"],
  "relevant_keywords": ["关键词1", "关键词2"],
  "difficulty": "{dataset_type['difficulty']}",
  "expected_behavior": "正常回答/拒答/说明无直接答案",
  "multi_hop_reasoning_chain": "多跳推理的推理链（非多跳类留空）",
  "safety_notes": "安全注意事项（非安全类留空）",
  "notes": "备注"
}}

## 质量要求（非常重要）

1. **问题多样性**：{batch_size} 条问题必须完全不同，不能重复或高度相似。每个问题要从不同角度、不同产品、不同场景出发。
2. **ground_truth 必须具体准确**：
   - 禁止模糊回答（如"请参考用户手册"、"具体步骤请查阅文档"）
   - 必须包含具体的数字、步骤、参数、操作方法
   - 操作步骤类必须列出完整的 1/2/3/4 步骤
   - 事实查询类必须给出明确的答案（如具体温度、具体型号、具体时间）
3. **relevant_doc_ids 必须准确**：填写知识库中真实存在的文档标题，不能编造。
4. **问题要自然**：符合真实用户的提问习惯，不要太书面化或像考试题。
5. **覆盖不同产品**：尽量覆盖知识库中的不同产品/设备，不要集中在某一个产品上。

只输出 JSON 数组，不要任何其他文字、解释或 markdown 代码块标记。
生成 {batch_size} 条，sample_id 从 TT-{start_id:03d} 开始连续编号。
"""

    return prompt


# ====================================================================
# 3. 调用 LLM 生成测试集
# ====================================================================
def generate_with_llm(prompt: str, max_retries: int = 3) -> List[Dict]:
    """调用 qwen-max 生成测试集，带重试和 JSON 解析"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    for attempt in range(max_retries):
        try:
            print(f"    调用 qwen-max (第{attempt+1}次)...")
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GEN_TEMPERATURE,
                max_tokens=8192,
            )
            content = response.choices[0].message.content.strip()

            # 清理可能的 markdown 标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 解析 JSON
            data = json.loads(content)
            if isinstance(data, list):
                print(f"    成功生成 {len(data)} 条")
                return data
            else:
                print(f"    返回不是数组，重试...")
        except json.JSONDecodeError as e:
            print(f"    JSON 解析失败: {e}，重试...")
        except Exception as e:
            print(f"    调用失败: {e}，重试...")
        time.sleep(2)

    print(f"    多次重试失败，跳过此批次")
    return []


# ====================================================================
# 4. 主流程
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="生成 ThinkTank 测试集")
    parser.add_argument("--limit", type=int, default=None, help="只生成前N条")
    parser.add_argument("--dry-run", action="store_true", help="只拉取数据，不调用LLM")
    parser.add_argument("--resume", action="store_true", help="从已有文件继续生成")
    args = parser.parse_args()

    print("=" * 60)
    print("ThinkTank 测试集生成脚本")
    print("=" * 60)
    print(f"目标总数: {TOTAL_COUNT} 条")
    print(f"生成模型: {GEN_MODEL}")
    print(f"输出文件: {OUTPUT_FILE}")
    print()

    # 1. 拉取知识库数据
    kb_data = fetch_knowledge_base()
    print()

    if args.dry_run:
        print("[dry-run] 数据拉取完成，不调用LLM")
        print(f"  商品名: {len(kb_data['item_names'])} 个")
        print(f"  文档标题: {len(kb_data['doc_titles'])} 个")
        print(f"  有内容的文档: {len([t for t, c in kb_data['doc_contents'].items() if c])} 个")
        return

    # 2. 加载已有结果（如果是 resume 模式）
    all_samples = []
    start_id = 4  # 从 TT-004 开始（沿用现有编号）
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        start_id = 4 + len(all_samples)
        print(f"[resume] 已加载 {len(all_samples)} 条，从 TT-{start_id:03d} 继续")

    # 3. 按类型分批生成
    print(f"\n[2/4] 开始生成测试集...")
    generated_count = 0

    for dtype in DATASET_TYPES:
        count = dtype["count"]
        if args.limit and generated_count >= args.limit:
            break

        batch_count = min(count, args.limit - generated_count) if args.limit else count
        if batch_count <= 0:
            continue

        print(f"\n  【{dtype['type']}】目标 {batch_count} 条 (难度: {dtype['difficulty']})")

        # 分批生成，每批最多 8 条（避免上下文过长）
        batch_size = min(8, batch_count)
        batches = (batch_count + batch_size - 1) // batch_size

        for b in range(batches):
            current_batch_size = min(batch_size, batch_count - b * batch_size)
            current_start_id = start_id + generated_count

            print(f"    批次 {b+1}/{batches}: {current_batch_size} 条 (ID: TT-{current_start_id:03d}~)")

            prompt = build_generation_prompt(dtype, kb_data, current_batch_size, current_start_id)
            samples = generate_with_llm(prompt)

            # 校验和清洗
            valid_samples = []
            seen_questions = set()  # 去重用
            for s in samples:
                # 确保必填字段存在
                if "question" not in s or "ground_truth" not in s:
                    continue
                # 去重：问题高度相似则跳过
                q = s["question"].strip().lower()
                if q in seen_questions:
                    print(f"    跳过重复问题: {s['question'][:30]}...")
                    continue
                seen_questions.add(q)
                # ground_truth 质量检查：太短或太模糊则标记
                gt = s.get("ground_truth", "")
                if len(gt) < 10 or "请参考" in gt or "查阅文档" in gt or "具体请" in gt:
                    s["notes"] = (s.get("notes", "") + " [ground_truth质量待审核]").strip()
                # 补全缺失字段
                s.setdefault("sample_id", f"TT-{current_start_id + len(valid_samples):03d}")
                s.setdefault("category", "")
                s.setdefault("question_type", dtype["type"])
                s.setdefault("relevant_doc_ids", [])
                s.setdefault("relevant_keywords", [])
                s.setdefault("difficulty", dtype["difficulty"])
                s.setdefault("expected_behavior", "拒答" if "拒答" in dtype["type"] else "正常回答")
                s.setdefault("multi_hop_reasoning_chain", "")
                s.setdefault("safety_notes", "")
                s.setdefault("notes", "")
                valid_samples.append(s)

            all_samples.extend(valid_samples)
            generated_count += len(valid_samples)
            print(f"    本批次有效 {len(valid_samples)} 条，累计 {generated_count} 条")

            # 保存中间结果（强制 UTF-8）
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)

            time.sleep(1)  # 避免 API 限流

    # 4. 统计和保存
    print(f"\n[3/4] 生成完成，统计结果...")
    type_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for s in all_samples:
        type_counts[s.get("question_type", "未知")] += 1
        difficulty_counts[s.get("difficulty", "未知")] += 1

    print(f"\n  总样本数: {len(all_samples)}")
    print(f"\n  按类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c} 条")
    print(f"\n  按难度分布:")
    for d, c in sorted(difficulty_counts.items()):
        print(f"    {d}: {c} 条")

    # 最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n[4/4] 测试集已保存到: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
