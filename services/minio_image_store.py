import logging
import os
import re
from typing import List, Tuple, Dict
from minio.deleteobjects import DeleteObject

from config.minio_config import minio_config
from utils.minio_utils import get_minio_client


class MinioImageStore:
    """MinIO图片存储服务：清理旧数据、批量上传、返回URL、替换MD引用"""

    def __init__(self):
        self.client = get_minio_client()
        self.bucket_name = minio_config.bucket_name
        self.img_dir = minio_config.img_dir
        self.logger = logging.getLogger(__name__)

    def upload_images(self, doc_stem: str, target_images: List[Tuple]) -> Dict[str, str]:
        """批量上传图片，返回 {文件名: URL}，上传失败为None"""
        upload_dir = f"{self.img_dir}/{doc_stem}".replace(" ", "")
        self._clean_directory(upload_dir)
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            urls[img_file] = self._upload(img_path, object_name)
        return urls

    def _clean_directory(self, prefix: str):
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            #DeleteObject(obj.object_name)构造一个待删除项的数据对象
            delete_list = [DeleteObject(obj.object_name) for obj in objects]
            if delete_list:
                errors = self.client.remove_objects(self.bucket_name, delete_list)
                for err in errors:
                    self.logger.error(f"删除失败：{err}")
        except Exception as e:
            self.logger.error(f"清理minio目录失败：{e}")

    def _upload(self, img_path: str, object_name: str) -> str | None:
        try:
            self.client.fput_object(
                bucket_name=self.bucket_name,
                file_path=img_path,
                object_name=object_name,
                content_type=f"image/{os.path.splitext(img_path)[1][1:]}"
            )
            base_url = f"http://{minio_config.endpoint}/{self.bucket_name}"
            return f"{base_url}/{object_name}"
        except Exception as e:
            self.logger.error(f"图片上传MinIO失败：{img_path}，错误：{e}")
            return None

    @staticmethod
    def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """合并摘要和URL，过滤上传失败的图片，返回 {文件名: (摘要, URL)}"""
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    @staticmethod
    def replace_in_md(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """将MD中本地图片引用替换为MinIO远程引用+摘要描述"""
        for img_file, (summary, new_url) in image_info.items():
            # [^\]]* 限制方括号内不能含]，[^)]* 限制括号内不能含)，避免跨图片引用匹配
            pattern = re.compile(r"!\[[^\]]*\]\([^)]*" + re.escape(img_file) + r"[^)]*\)")
            #pattern.sub(替换规则, 被处理的字符串)
            md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        return md_content