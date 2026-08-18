
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from datasets import load_dataset
import shutil

# 完整加载，包含全部元数据category、pdf路径
ds = load_dataset("opendatalab/OmniDocBench", split="train")

print(f"总样本数：{len(ds)}")

manual_samples = []
for item in ds:
    cat = item.get("category", "")
    if cat == "Manual":
        manual_samples.append(item)

print(f"筛选Manual手册数量：{len(manual_samples)}")

OUT_PDF_FOLDER = r"D:\datasets\OmniDocBench_ProductPDF"
os.makedirs(OUT_PDF_FOLDER, exist_ok=True)

for sample in manual_samples:
    src_pdf = sample["pdf"]
    if not os.path.exists(src_pdf):
        continue
    dst_pdf = os.path.join(OUT_PDF_FOLDER, os.path.basename(src_pdf))
    shutil.copy2(src_pdf, dst_pdf)

print(f"PDF输出目录：{OUT_PDF_FOLDER}")