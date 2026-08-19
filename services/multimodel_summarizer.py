import base64
import io
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Deque
from PIL import Image

from config.lm_config import lm_config
from utils.llm_utils import get_llm_client


class MultimodalSummarizer:
    """多模态摘要服务：调用大模型生成图片摘要，带API滑动窗口限流，支持线程池并发"""

    def __init__(self, requests_per_minute: int = 15, max_side: int = 1568, quality: int = 85, max_workers: int = 5):
        self.requests_per_minute = requests_per_minute
        self.max_side = max_side
        self.quality = quality
        self.max_workers = max_workers

    def generate_summaries(self, doc_stem: str, target_images: List[Tuple]) -> Dict[str, str]:
        """批量为图片生成内容摘要，线程池并发，返回 {文件名: 摘要}"""
        lock = threading.Lock()
        request_deque: Deque[float] = deque()
        summaries = {}

        def _process(img_file: str, image_path: str, context: Tuple[str, str]):
            # 限流必须在锁内，多线程共享同一个滑动窗口
            with lock:
                self._apply_rate_limit(request_deque)
            summary = self._summarize_image(image_path, doc_stem, context)
            return img_file, summary

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_process, *img) for img in target_images]
            for future in as_completed(futures):
                img_file, summary = future.result()
                summaries[img_file] = summary

        return summaries

    def _apply_rate_limit(self, request_times: Deque[float], window_seconds: int = 60):
        current_time = time.time()
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()
        if len(request_times) >= self.requests_per_minute:
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                time.sleep(sleep_duration)
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()
        request_times.append(current_time)

    def _encode_image(self, image_path: str):
        """压缩大图并转base64，返回 (base64_str, mime_type)"""
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, self.max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"

    def _summarize_image(self, image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
        base64_image, mime = self._encode_image(image_path)
        try:
            chat_model = get_llm_client(model=lm_config.vl_model)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f'这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。'},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}}
                ]
            }]
            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n", "")
        except Exception as e:
            return "图片描述"