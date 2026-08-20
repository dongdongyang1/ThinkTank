import json
import logging
import os
import shutil
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from services.image_scanner import ImageScanner
from services.minio_image_store import MinioImageStore
from typing import Tuple

from services.multimodel_summarizer import MultimodalSummarizer


class NodeMDImg(BaseNode):
    """MarkDown图片处理节点：编排图片扫描→摘要生成→MinIO上传→MD替换"""
    name = "c_node_md_img"

    def process(self, state: ImportGraphState):
        # PDF解析失败则跳过
        if state.get("pdf_parse_error"):
            self.logger.warning(f"跳过本节点，PDF解析失败:{state['pdf_parse_error']}")
            return state

        # 1. 获取MD核心信息
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        # === 缓存检查 ===
        cache_dir = md_path_obj.parent.parent.parent / ".img_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{md_path_obj.stem}_cached.md"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            self.logger.info(f"命中图片处理缓存，跳过图片理解：{cache_file.name}")
            with open(cache_file, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            state["md_path"] = str(cache_file.absolute())
            return state
        # === 缓存检查结束 ===

        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        # 2. 扫描图片（委托服务）
        scanner = ImageScanner(self.config.image_extensions)
        target_images = scanner.scan(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到MD中引用了图片，跳过图片处理")
            return state

        # 3. 调用多模态大模型生成摘要（委托服务）
        summarizer = MultimodalSummarizer(requests_per_minute=self.config.requests_per_minute)
        summaries = summarizer.generate_summaries(md_path_obj.stem, target_images)

        # 4. 上传MinIO并替换MD路径（委托服务）
        store = MinioImageStore()
        urls = store.upload_images(md_path_obj.stem, target_images)
        image_info = store.merge_summary_and_url(summaries, urls)
        new_md_content = store.replace_in_md(md_content, image_info)

        # 5. 备份并保存新MD
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        # === 保存缓存 ===
        shutil.copy2(new_md_file_name, cache_file)
        self.logger.info(f"图片处理结果已写入缓存：{cache_file.name}")
        # === 保存缓存结束 ===

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(field_name='md_path', expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"MD文件{md_path_obj.name}不存在")
        md_content = state.get("md_content")
        images_dir = md_path_obj.parent / "images"
        return md_content, md_path_obj, images_dir

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        self.logger.info(f"处理后MD文件已保存：{new_md_file_name}")
        return new_md_file_name


if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content
    }

    # 执行核心处理流程
    node_md_img = NodeMDImg()
    result = node_md_img(init_state)
    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))