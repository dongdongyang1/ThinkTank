import os
import re
from pathlib import Path
from typing import List,Optional, Tuple


class ImageScanner:
    """图片扫描服务：扫描目录，筛选MD中实际引用的支持格式图片"""
    def __init__(self,image_extensions : set):
        self.image_extensions = image_extensions

    def scan(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，返回 [(文件名, 完整路径, (上文, 下文)), ...]
        """
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in self.image_extensions:
                continue
            img_path = str(images_dir / image_file)
            # 查找图片在MD中的上下文
            context = self._find_image_in_md(md_content, image_file)
            if not context:
                continue
            target_images.append((image_file, img_path, context))
        return target_images

    @staticmethod
    def _find_image_in_md(md_content: str, image_file: str, context_len: int = 100) -> Optional[Tuple[str, str]]:
        #re.escape(image_file)：对图片文件名转义，文件名里如果有 . \ * ? 这类正则特殊符号，不会被当成正则语法，当作普通文本匹配
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text, post_text