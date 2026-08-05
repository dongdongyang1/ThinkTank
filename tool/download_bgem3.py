from modelscope import snapshot_download

model_dir = snapshot_download("BAAI/bge-m3",cache_dir="D:\modelscope_cache\models")
print(f"模型已下载到: {model_dir}")

