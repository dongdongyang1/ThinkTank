import time
from typing import List

import dashscope
from dotenv import load_dotenv
from config.reranker_config import reranker_config

load_dotenv()

dashscope.api_key = reranker_config.text_rerank_api_key


def _rerank_once(query: str, documents: List[str]) -> list:
    response = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct
    )
    status_code = response.get("status_code")
    if status_code != 200:
        raise RuntimeError(f"DashScope rerank 调用失败: {response.get('message')}")
    results = response.output.get("results", [])
    scores = [0.0] * len(documents)
    for item in results:
        scores[int(item.get("index"))] = float(item.get("relevance_score"))
    return scores


def rerank_documents(query: str, documents: List[str]) -> list[float]:
    """带重试的rerank调用：dashscope SDK无timeout参数，用重试+退避兜底瞬时故障"""
    last_exc = None
    for attempt in range(3):
        try:
            return _rerank_once(query, documents)
        except Exception as e:
            last_exc = e
            time.sleep(2 * (attempt + 1))  # 2s、4s 退避
    raise last_exc
