"""
图片处理服务单元测试：覆盖图片扫描、MD中图片引用查找、MinIO路径替换
"""

import os
import tempfile
from pathlib import Path

import pytest

from services.image_scanner import ImageScanner
from services.minio_image_store import MinioImageStore


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def scanner():
    return ImageScanner(image_extensions={".jpg", ".jpeg", ".png", ".gif"})


class TestImageScanner:
    def test_find_image_in_md_standard(self, scanner):
        """标准Markdown图片引用能被找到"""
        md = "这是正文 ![描述](images/test.jpg) 后续内容"
        result = scanner._find_image_in_md(md, "test.jpg")
        assert result is not None
        assert "这是正文" in result[0]
        assert "后续内容" in result[1]

    def test_find_image_in_md_not_found(self, scanner):
        """不存在的图片返回None"""
        md = "![描述](images/other.jpg)"
        assert scanner._find_image_in_md(md, "missing.jpg") is None

    def test_find_image_with_special_chars(self, scanner):
        """图片名含特殊字符（正则元字符）也能匹配"""
        md = "![图](images/test(1).jpg)"
        result = scanner._find_image_in_md(md, "test(1).jpg")
        assert result is not None

    def test_scan_filters_by_extension(self, scanner, temp_dir):
        """不支持的格式被过滤"""
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        (img_dir / "a.jpg").write_bytes(b"fake")
        (img_dir / "b.tiff").write_bytes(b"fake")  # 不支持的格式
        (img_dir / "c.png").write_bytes(b"fake")
        md = "![a](images/a.jpg) ![c](images/c.png)"
        results = scanner.scan(md, img_dir)
        names = [r[0] for r in results]
        assert "a.jpg" in names
        assert "c.png" in names
        assert "b.tiff" not in names

    def test_scan_filters_unreferenced(self, scanner, temp_dir):
        """MD中未引用的图片被跳过"""
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        (img_dir / "referenced.jpg").write_bytes(b"fake")
        (img_dir / "orphan.jpg").write_bytes(b"fake")
        md = "![图](images/referenced.jpg)"
        results = scanner.scan(md, img_dir)
        names = [r[0] for r in results]
        assert names == ["referenced.jpg"]

    def test_scan_empty_directory(self, scanner, temp_dir):
        """空图片目录返回空列表"""
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        assert scanner.scan("some md", img_dir) == []


class TestMinioImageStoreReplace:
    """测试MD图片引用替换（纯逻辑，不连MinIO）"""

    def test_replace_single_image(self):
        """单张图片引用被替换为MinIO URL+摘要"""
        md = "正文 ![旧描述](images/old.jpg) 结尾"
        image_info = {"old.jpg": ("新摘要", "http://minio/bucket/img/old.jpg")}
        result = MinioImageStore.replace_in_md(md, image_info)
        assert "![新摘要](http://minio/bucket/img/old.jpg)" in result
        assert "旧描述" not in result

    def test_replace_multiple_images(self):
        """多张图片同时替换"""
        md = "![a](images/a.jpg) ![b](images/b.png)"
        image_info = {
            "a.jpg": ("摘要A", "http://minio/a.jpg"),
            "b.png": ("摘要B", "http://minio/b.png")
        }
        result = MinioImageStore.replace_in_md(md, image_info)
        assert "![摘要A](http://minio/a.jpg)" in result
        assert "![摘要B](http://minio/b.png)" in result

    def test_replace_preserves_other_content(self):
        """非图片内容保持不变"""
        md = "# 标题\n\n正文段落\n\n![图](images/x.jpg)\n\n更多正文"
        image_info = {"x.jpg": ("摘要", "http://minio/x.jpg")}
        result = MinioImageStore.replace_in_md(md, image_info)
        assert "# 标题" in result
        assert "正文段落" in result
        assert "更多正文" in result

    def test_replace_with_special_chars_in_url(self):
        """URL含特殊字符不影响替换"""
        md = "![d](images/pic.jpg)"
        image_info = {"pic.jpg": ("描述", "http://minio/bucket/path%20with%20spaces.jpg")}
        result = MinioImageStore.replace_in_md(md, image_info)
        assert "http://minio/bucket/path%20with%20spaces.jpg" in result

    def test_merge_summary_and_url_filters_none(self):
        """上传失败（URL为None）的图片被过滤"""
        summaries = {"a.jpg": "摘要A", "b.jpg": "摘要B"}
        urls = {"a.jpg": "http://minio/a.jpg", "b.jpg": None}
        result = MinioImageStore.merge_summary_and_url(summaries, urls)
        assert "a.jpg" in result
        assert "b.jpg" not in result
        assert result["a.jpg"] == ("摘要A", "http://minio/a.jpg")

    def test_merge_summary_and_url_empty(self):
        """空输入返回空字典"""
        assert MinioImageStore.merge_summary_and_url({}, {}) == {}
