"""
split_pdf 工具单元测试：覆盖PDF切分、页数读取、多分片MD/图片合并
"""

import os
import tempfile
from pathlib import Path

import pymupdf
import pytest

from tool.split_pdf import split_pdf, merge_parts, get_pdf_page_count, MAX_PAGES_PER_FILE


@pytest.fixture
def temp_dir():
    """提供临时目录，测试结束自动清理"""
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _create_test_pdf(path: Path, num_pages: int = 10):
    """创建指定页数的测试PDF"""
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 50), f"Test Page {i + 1}")
    doc.save(path)
    doc.close()


class TestGetPdfPageCount:
    def test_normal_pdf(self, temp_dir):
        pdf = temp_dir / "test.pdf"
        _create_test_pdf(pdf, num_pages=5)
        assert get_pdf_page_count(pdf) == 5

    def test_single_page(self, temp_dir):
        pdf = temp_dir / "single.pdf"
        _create_test_pdf(pdf, num_pages=1)
        assert get_pdf_page_count(pdf) == 1


class TestSplitPdf:
    def test_split_under_limit(self, temp_dir):
        """页数不超过限制时，只切1片"""
        pdf = temp_dir / "small.pdf"
        _create_test_pdf(pdf, num_pages=10)
        parts = split_pdf(pdf, temp_dir, max_pages=200)
        assert len(parts) == 1
        assert get_pdf_page_count(parts[0]) == 10

    def test_split_exact_multiple(self, temp_dir):
        """页数刚好是限制的整数倍"""
        pdf = temp_dir / "exact.pdf"
        _create_test_pdf(pdf, num_pages=10)
        parts = split_pdf(pdf, temp_dir, max_pages=5)
        assert len(parts) == 2
        assert get_pdf_page_count(parts[0]) == 5
        assert get_pdf_page_count(parts[1]) == 5

    def test_split_with_remainder(self, temp_dir):
        """页数不是整数倍，最后一片是余数"""
        pdf = temp_dir / "remainder.pdf"
        _create_test_pdf(pdf, num_pages=12)
        parts = split_pdf(pdf, temp_dir, max_pages=5)
        assert len(parts) == 3
        assert get_pdf_page_count(parts[0]) == 5
        assert get_pdf_page_count(parts[1]) == 5
        assert get_pdf_page_count(parts[2]) == 2

    def test_default_max_pages_constant(self, temp_dir):
        """默认使用 MAX_PAGES_PER_FILE 常量"""
        pdf = temp_dir / "default.pdf"
        _create_test_pdf(pdf, num_pages=10)
        parts = split_pdf(pdf, temp_dir)
        assert len(parts) == 1
        assert MAX_PAGES_PER_FILE == 200

    def test_output_filenames(self, temp_dir):
        """切分文件命名规范：原文件名_part{N}.pdf"""
        pdf = temp_dir / "mydoc.pdf"
        _create_test_pdf(pdf, num_pages=12)
        parts = split_pdf(pdf, temp_dir, max_pages=5)
        assert parts[0].name == "mydoc_part1.pdf"
        assert parts[1].name == "mydoc_part2.pdf"
        assert parts[2].name == "mydoc_part3.pdf"


class TestMergeParts:
    def _create_part_md(self, base_dir: Path, stem: str, content: str, images: dict = None):
        """创建一个模拟的分片MD目录结构"""
        part_dir = base_dir / stem
        part_dir.mkdir(parents=True, exist_ok=True)
        md_path = part_dir / f"{stem}.md"
        md_path.write_text(content, encoding="utf-8")
        if images:
            img_dir = part_dir / "images"
            img_dir.mkdir(exist_ok=True)
            for name, data in images.items():
                (img_dir / name).write_bytes(data)
        return md_path

    def test_merge_two_parts_content(self, temp_dir):
        """两个分片内容按顺序合并，中间有---分隔"""
        md1 = self._create_part_md(temp_dir, "doc_part1", "# Part1\nHello")
        md2 = self._create_part_md(temp_dir, "doc_part2", "# Part2\nWorld")
        result = merge_parts([md1, md2], temp_dir, "doc")
        merged = result.read_text(encoding="utf-8")
        assert "# Part1" in merged
        assert "# Part2" in merged
        assert "---" in merged
        assert merged.index("# Part1") < merged.index("# Part2")

    def test_merge_with_images_rename(self, temp_dir):
        """图片重命名加part前缀，MD中路径同步更新"""
        img_data = b"\xff\xd8\xff\xe0fakejpg"
        md1 = self._create_part_md(
            temp_dir, "doc_part1",
            "![图1](images/img_a.jpg)\n内容A",
            images={"img_a.jpg": img_data}
        )
        md2 = self._create_part_md(
            temp_dir, "doc_part2",
            "![图2](images/img_a.jpg)\n内容B",  # 同名图片，验证不冲突
            images={"img_a.jpg": img_data}
        )
        result = merge_parts([md1, md2], temp_dir, "doc")
        merged = result.read_text(encoding="utf-8")

        # 两个同名图片被重命名为 part1_img_a.jpg 和 part2_img_a.jpg
        assert "part1_img_a.jpg" in merged
        assert "part2_img_a.jpg" in merged

        # 最终images目录有两张不重名的图片
        final_images = result.parent / "images"
        assert (final_images / "part1_img_a.jpg").exists()
        assert (final_images / "part2_img_a.jpg").exists()

    def test_merge_output_path(self, temp_dir):
        """合并后的MD路径为 output_dir/pdf_stem/pdf_stem.md"""
        md1 = self._create_part_md(temp_dir, "doc_part1", "content1")
        result = merge_parts([md1], temp_dir, "doc")
        assert result.name == "doc.md"
        assert result.parent.name == "doc"

    def test_merge_cleans_up_part_dirs(self, temp_dir):
        """合并完成后清理分片解压目录"""
        md1 = self._create_part_md(temp_dir, "doc_part1", "content1")
        md2 = self._create_part_md(temp_dir, "doc_part2", "content2")
        merge_parts([md1, md2], temp_dir, "doc")
        assert not (temp_dir / "doc_part1").exists()
        assert not (temp_dir / "doc_part2").exists()
