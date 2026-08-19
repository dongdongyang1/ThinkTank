import pymupdf
from pathlib import Path
import tempfile

def pre_compress_pdf_large_image(origin_pdf: Path) -> Path:
    """
    预处理PDF，对超高DPI/超大像素图片做降采样，解决MinerU DecompressionBomb失败
    返回临时压缩后的PDF路径；原始文件完全不动
    """
    doc = pymupdf.open(origin_pdf)
    # 将所有DPI>120的图片，降采样到96DPI，JPEG质量70；文字、矢量图不受影响
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
    # 生成临时文件
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp_fd.name)
    tmp_fd.close()
    doc.ez_save(tmp_path)
    doc.close()
    return tmp_path