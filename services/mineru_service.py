"""
MinerU API 服务层：封装PDF上传、解析轮询、结果下载解压

节点只调用本服务的公开方法，不直接接触 requests / MinerU API细节
"""

import hashlib
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
import requests

from config.mineru_config import mineru_config


class MinerUService:
    """MinerU PDF结构化解析服务，支持文件hash幂等缓存"""

    def __init__(self, timeout_seconds: int = 1200, poll_interval: int = 3):
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mineru_config.api_token}"
        })
        # 幂等缓存目录：用文件hash做key，Celery重试时不重复消耗MinerU额度
        self.cache_dir = Path(os.getenv("MINERU_CACHE_DIR", "./.mineru_cache")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: Path, task_id: str = "") -> str:
        """
        上传PDF至MinerU并轮询解析，返回结果ZIP包下载链接
        :param pdf_path: PDF文件路径
        :param task_id: 任务ID（用于日志和进度上报）
        :return: 解析结果ZIP包下载URL
        """
        # 1. 获取上传链接
        upload_url = f"{mineru_config.base_url}/file-urls/batch"
        data = {"files": [{"name": pdf_path.name}], "model_version": "vlm"}
        response = self.session.post(upload_url, json=data, timeout=(10, 30))
        if response.status_code != 200:
            raise RuntimeError(f"获取上传链接失败：状态码{response.status_code}")
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"获取上传链接失败：{result}")

        signal_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]
        self.logger.info(f"[MinerU] 获取上传链接成功，batch_id={batch_id}")

        # 2. 上传文件（预签名URL不能带额外header）
        with open(pdf_path, "rb") as f:
            res_upload = requests.put(signal_url, data=f, timeout=(10, 600))
            if res_upload.status_code != 200:
                raise RuntimeError(f"文件上传失败：状态码{res_upload.status_code}，响应体：{res_upload.text}")
        self.logger.info(f"[MinerU] 文件上传成功：{pdf_path.name}")

        # 3. 轮询解析状态
        poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        start_time = time.time()
        consecutive_errors = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                raise TimeoutError(f"MinerU解析超时：{self.timeout_seconds}秒，batch_id={batch_id}")

            try:
                res_poll = self.session.get(poll_url, timeout=10)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise RuntimeError(f"MinerU轮询连续网络异常：{e}")
                time.sleep(self.poll_interval)
                continue

            if res_poll.status_code != 200:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise RuntimeError(f"MinerU轮询HTTP失败：状态码{res_poll.status_code}")
                time.sleep(self.poll_interval)
                continue

            poll_data = res_poll.json()
            if poll_data.get("code") != 0:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise RuntimeError(f"MinerU轮询业务错误：{poll_data}")
                time.sleep(self.poll_interval)
                continue

            consecutive_errors = 0
            extract_results = poll_data["data"]["extract_result"]
            if not extract_results:
                time.sleep(self.poll_interval)
                continue

            result_item = extract_results[0]
            data_state = result_item["state"]

            if data_state == "done":
                self.logger.info(f"[MinerU] 解析完成，耗时{int(elapsed)}s，batch_id={batch_id}")
                return result_item["full_zip_url"]
            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误")
                raise RuntimeError(f"MinerU解析失败：batch_id={batch_id}，错误：{err_msg}")
            else:
                self.logger.info(f"[MinerU] 解析中... 已耗时{int(elapsed)}s，状态={data_state}")
                time.sleep(self.poll_interval)

    def download_and_extract(self, zip_url: str, output_dir: Path, pdf_stem: str, task_id: str = "") -> str:
        """
        下载MinerU结果ZIP包并解压，提取MD文件
        :param zip_url: ZIP包下载链接
        :param output_dir: 输出根目录
        :param pdf_stem: PDF文件名（不含后缀），用于命名解压目录和MD文件
        :param task_id: 任务ID
        :return: MD文件绝对路径
        """
        # 1. 下载ZIP
        zip_save_path = output_dir / f"{pdf_stem}_result.zip"
        with requests.get(zip_url, timeout=(10, 600), stream=True) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"ZIP下载失败：状态码{resp.status_code}")
            with open(zip_save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        self.logger.info(f"[MinerU] ZIP下载成功：{zip_save_path}")

        # 2. 清理旧解压目录
        extract_target_dir = output_dir / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 3. 解压
        with zipfile.ZipFile(zip_save_path, "r") as zf:
            zf.extractall(extract_target_dir)
        self.logger.info(f"[MinerU] ZIP解压完成：{extract_target_dir}")

        # 4. 找到MD文件并重命名为 pdf_stem.md（MinerU输出的MD文件名不固定，用glob找）
        md_files = list(extract_target_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"MinerU输出中未找到MD文件，目录内容：{list(extract_target_dir.iterdir())}")
        if len(md_files) > 1:
            self.logger.warning(f"MinerU输出中有多个MD文件，使用第一个：{[f.name for f in md_files]}")
        target_md = md_files[0]
        original_name = target_md.name
        new_md_path = target_md.with_name(f"{pdf_stem}.md")
        target_md.rename(new_md_path)
        self.logger.info(f"[MinerU] MD重命名完成：{original_name} -> {pdf_stem}.md")

        return str(new_md_path.absolute())

    # ========== 幂等缓存：避免Celery重试重复消耗MinerU额度 ==========

    def _get_file_hash(self, pdf_path: Path) -> str:
        """计算PDF文件MD5（前16位），作为缓存key"""
        h = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def parse_pdf_to_md(self, pdf_path: Path, output_dir: Path, task_id: str = "", output_stem: str = None) -> str:
        """
        高层方法：带幂等缓存的PDF解析→MD提取
        同一文件内容（hash相同）只调用一次MinerU，后续直接从缓存取结果
        :param pdf_path: PDF文件路径（可能是压缩后的临时文件）
        :param output_dir: 输出根目录
        :param task_id: 任务ID
        :param output_stem: 输出文件名（不含后缀），传None则用pdf_path.stem
        :return: MD文件绝对路径
        """
        if output_stem is None:
            output_stem = pdf_path.stem

        file_hash = self._get_file_hash(pdf_path)
        cache_entry = self.cache_dir / file_hash
        cached_md = cache_entry / f"{output_stem}.md"

        # 1. 缓存命中：直接复制缓存结果到输出目录，不调用MinerU
        if cached_md.exists():
            self.logger.info(f"[MinerU] 缓存命中，跳过API调用，hash={file_hash}")
            target_dir = output_dir / output_stem
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(cache_entry, target_dir)
            return str((target_dir / f"{output_stem}.md").absolute())

        # 2. 缓存未命中：正常调用MinerU
        self.logger.info(f"[MinerU] 缓存未命中，调用API解析，hash={file_hash}")
        zip_url = self.parse_pdf(pdf_path, task_id)
        md_path = self.download_and_extract(zip_url, output_dir, output_stem, task_id)

        # 3. 写入缓存（复制整个解压目录）
        try:
            source_dir = Path(md_path).parent
            if cache_entry.exists():
                shutil.rmtree(cache_entry)
            shutil.copytree(source_dir, cache_entry)
            self.logger.info(f"[MinerU] 解析结果已缓存，hash={file_hash}")
        except Exception as e:
            self.logger.warning(f"[MinerU] 缓存写入失败（不影响主流程）：{e}")

        return md_path
