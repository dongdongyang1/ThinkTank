"""
定义minio_utils客户端工具:初始化 MinIO 对象存储客户端，
自动创建桶、设置公开只读权限，后续上传、下载图片直接调用get_minio_client()拿客户端即可使用
"""
import json

from minio import Minio
from config.minio_config import minio_config

try:
    minio_client = Minio(
        endpoint = minio_config.endpoint,
        access_key = minio_config.access_key,
        secret_key = minio_config.secret_key,
        secure = False
    )
    # secure=False 是否启用HTTPS加密连接；False=用HTTP，True=用HTTPS；本地/内网部署一律写False
    if not minio_client.bucket_exists(minio_config.bucket_name):
        minio_client.make_bucket(minio_config.bucket_name)

    # 设置存储桶策略为 Public Read (只读权限开放给匿名用户)
    # 这样前端可以直接通过 URL 访问图片，而不需要预签名 URL
    # Python 字典
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],   #文件下载
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"]
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetBucketLocation"],  #桶位置查询权限
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}"]
            }
        ]
    }
    #入参要求 JSON 字符串
    minio_client.set_bucket_policy(minio_config.bucket_name,json.dumps(policy))

except Exception as e:
    print(f"Minio failed:{e}")
    minio_client = None

def get_minio_client():
    return minio_client
