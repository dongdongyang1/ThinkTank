import logging
import pymupdf
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


def pre_compress_pdf_large_image(origin_pdf: Path) -> Path:
    """
    预处理PDF，对超高DPI/超大像素图片做降采样，解决MinerU DecompressionBomb失败
    返回临时压缩后的PDF路径；原始文件完全不动

    兼容带 alpha 透明通道的图片：
    1. 先尝试有损压缩（JPEG，体积小，但不支持 alpha 通道）
    2. 失败则回退无损压缩（PNG，支持 alpha 通道，但体积大）
    3. 再失败则跳过压缩，直接用原始PDF（兜底，不阻塞主流程）
    """
    # ========== 尝试1：有损压缩（JPEG） ==========
    doc = pymupdf.open(origin_pdf)
    try:
        doc.rewrite_images(
            dpi_threshold=120,
            dpi_target=96,
            quality=70,
            lossy=True,
            lossless=True,
            bitonal=True,
            color=True,
            gray=True,
            set_to_gray=False
        )
        logger.info(f"PDF图片压缩成功（有损JPEG）: {origin_pdf.name}")
    except Exception as e:
        # 图片带 alpha 通道导致 JPEG 转换失败，回退到无损压缩（PNG 支持 alpha）
        logger.warning(f"PDF有损压缩失败，回退无损压缩: {e}")
        doc.close()

        # ========== 尝试2：无损压缩（PNG） ==========
        doc = pymupdf.open(origin_pdf)
        try:
            doc.rewrite_images(
                dpi_threshold=120,
                dpi_target=96,
                quality=70,
                lossy=False,
                lossless=True,
                bitonal=True,
                color=True,
                gray=True,
                set_to_gray=False
            )
            logger.info(f"PDF图片压缩成功（无损PNG）: {origin_pdf.name}")
        except Exception as e2:
            # ========== 兜底：压缩都失败，直接用原始PDF ==========
            logger.warning(f"PDF图片压缩全部失败，跳过压缩，使用原始PDF: {e2}")
            doc.close()
            return origin_pdf

    # 生成临时文件
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp_fd.name)
    tmp_fd.close()
    doc.ez_save(tmp_path)
    doc.close()
    return tmp_path
