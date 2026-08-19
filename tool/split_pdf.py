"""
大PDF处理工具：切分 + 多分片MD/图片合并

纯文件操作，不依赖MinerU API，可独立测试和复用。
"""

import re
import shutil
from pathlib import Path
import pymupdf

# MinerU API单文件页数硬限制
MAX_PAGES_PER_FILE = 200


def get_pdf_page_count(pdf_path: Path) -> int:
    """获取PDF总页数"""
    doc = pymupdf.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


def split_pdf(pdf_path: Path, output_dir: Path, max_pages: int = MAX_PAGES_PER_FILE) -> list:
    """
    将PDF按max_pages页切分，返回分片Path列表

    Args:
        pdf_path: 原PDF路径
        output_dir: 切分文件输出目录
        max_pages: 每个分片最大页数，默认200

    Returns:
        分片PDF路径列表
    """
    doc = pymupdf.open(pdf_path)
    total = doc.page_count
    part_paths = []

    for i in range(0, total, max_pages):
        part_doc = pymupdf.open()
        end = min(i + max_pages, total)
        part_doc.insert_pdf(doc, from_page=i, to_page=end - 1)
        part_path = output_dir / f"{pdf_path.stem}_part{len(part_paths) + 1}.pdf"
        part_doc.save(part_path)
        part_doc.close()
        part_paths.append(part_path)

    doc.close()
    return part_paths


def merge_parts(part_md_paths: list, output_dir: Path, pdf_stem: str) -> Path:
    """
    合并多个分片的MD内容和图片：
    - 图片复制到统一images目录，重命名加part前缀避免冲突
    - MD中图片引用路径同步更新
    - 按分片顺序拼接MD内容，分片间用---分隔

    Args:
        part_md_paths: 各分片MD文件路径列表（Path对象）
        output_dir: 输出根目录
        pdf_stem: 原PDF文件名（不含后缀），用于命名最终目录和MD

    Returns:
        合并后的MD文件路径
    """
    # 1. 创建最终输出目录
    final_dir = output_dir / pdf_stem
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_images_dir = final_dir / "images"
    final_images_dir.mkdir(exist_ok=True)

    merged_content_parts = []

    for idx, part_md_path in enumerate(part_md_paths, start=1):
        # 2. 读取分片MD内容
        with open(part_md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 3. 处理该分片的images目录
        part_images_dir = part_md_path.parent / "images"
        if part_images_dir.exists():
            for img_file in part_images_dir.iterdir():
                if not img_file.is_file():
                    continue
                # 重命名：part{idx}_{原文件名}，避免不同分片图片重名
                new_img_name = f"part{idx}_{img_file.name}"
                dest_path = final_images_dir / new_img_name
                shutil.copy2(img_file, dest_path)
                # 替换MD中的图片引用路径
                escaped_name = re.escape(img_file.name)
                pattern = re.compile(r"(!\[[^\]]*\]\()[^)]*?" + escaped_name + r"(\))")
                md_content = pattern.sub(
                    lambda m: f"{m.group(1)}images/{new_img_name}{m.group(2)}",
                    md_content
                )

        # 4. 分片之间加分隔标记
        if idx > 1:
            merged_content_parts.append("\n\n---\n\n")
        merged_content_parts.append(md_content)

    # 5. 保存合并后的MD
    final_md_path = final_dir / f"{pdf_stem}.md"
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write("".join(merged_content_parts))

    # 6. 清理分片解压目录
    for part_md_path in part_md_paths:
        part_dir = part_md_path.parent
        if part_dir.exists() and part_dir != final_dir:
            shutil.rmtree(part_dir, ignore_errors=True)

    return final_md_path
