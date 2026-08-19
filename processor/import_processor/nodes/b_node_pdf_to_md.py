import logging
import shutil
import tempfile
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, PdfConversionError
from processor.import_processor.state import ImportGraphState
from services.mineru_service import MinerUService
from tool.pre_images import pre_compress_pdf_large_image
from tool.split_pdf import split_pdf, merge_parts, get_pdf_page_count
from utils.task_utils import set_task_result


class NodePDFToMD(BaseNode):
    """
    PDF转MarkDown节点：PDF结构化解析
    支持超过200页的大PDF：自动切分→逐片解析→合并MD与图片
    MinerU交互委托给 MinerUService
    """
    name = "b_node_pdf_to_md"

    def __init__(self):
        super().__init__()
        self.mineru = MinerUService(timeout_seconds=self.config.mineru_timeout_seconds)

    def process(self, state: ImportGraphState):
        try:
            self.logger.info("=====进入 b_node_pdf_to_md 节点 =====")
            pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

            page_count = get_pdf_page_count(pdf_path_obj)
            self.logger.info(f"PDF总页数：{page_count}")

            if page_count > self.config.mineru_max_pages:
                self.logger.info(f"PDF超过{self.config.mineru_max_pages}页，启动自动切分流程...")
                set_task_result(state["task_id"], "node_progress",
                                f"PDF共{page_count}页，超过限制，自动切分中...")
                md_path = self._process_large_pdf(pdf_path_obj, output_dir_obj, state["task_id"])
            else:
                compressed_pdf_path = pre_compress_pdf_large_image(pdf_path_obj)
                self.logger.info(f"完成PDF图片预处理：{compressed_pdf_path}")
                set_task_result(state["task_id"], "node_progress", "正在上传MinerU解析...")
                md_path = self.mineru.parse_pdf_to_md(compressed_pdf_path, output_dir_obj, state["task_id"])

            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            state["md_path"] = md_path
            state["md_content"] = md_content
            state["pdf_parse_error"] = None

        except (PdfConversionError, TimeoutError, StateFieldError, FileProcessingError, RuntimeError) as e:
            err_msg = f"PDF解析失败：{str(e)}"
            self.logger.error(err_msg, exc_info=True)
            state["pdf_parse_error"] = err_msg
            state["md_path"] = ""
            state["md_content"] = ""

        return state

    def _process_large_pdf(self, pdf_path_obj: Path, output_dir_obj: Path, task_id: str) -> str:
        """大PDF：切分→逐片解析→合并"""
        pdf_stem = pdf_path_obj.stem
        split_dir = Path(tempfile.mkdtemp(prefix="pdf_split_"))
        part_paths = split_pdf(pdf_path_obj, split_dir, max_pages=self.config.mineru_max_pages)
        self.logger.info(f"PDF已切分为{len(part_paths)}个分片")

        try:
            part_md_paths = []
            for idx, part_path in enumerate(part_paths, start=1):
                part_stem = f"{pdf_stem}_part{idx}"
                self.logger.info(f"正在解析第{idx}/{len(part_paths)}片：{part_path.name}")
                set_task_result(task_id, "node_progress", f"解析分片{idx}/{len(part_paths)}：{part_path.name}")

                compressed_path = pre_compress_pdf_large_image(part_path)
                part_md_path = self.mineru.parse_pdf_to_md(compressed_path, output_dir_obj, task_id)
                part_md_paths.append(Path(part_md_path))

            final_md_path = merge_parts(part_md_paths, output_dir_obj, pdf_stem)
            self.logger.info(f"大PDF合并完成：{final_md_path}")
            set_task_result(task_id, "node_progress", "大PDF分片解析与合并完成")
            return str(final_md_path.absolute())

        finally:
            shutil.rmtree(split_dir, ignore_errors=True)
            self.logger.info("已清理PDF切分临时目录")

    def _step_1_validate_paths(self, state: ImportGraphState):
        """校验PDF路径和输出目录"""
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_name", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF文件{pdf_path_obj.name}不存在")

        if not file_dir_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{file_dir_obj.absolute()}")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj


if __name__ == "__main__":
    setup_logging()
    init_state = {
        "task_id": "debug_test",
        "pdf_path": r"D:\hak180产品安全手册.pdf",
        "file_dir": r"D:\output"
    }
    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)
    logging.getLogger().info(str(result))
