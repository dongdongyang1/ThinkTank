import datetime
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any

import uvicorn
from starlette.responses import FileResponse
from fastapi import FastAPI, HTTPException, File, UploadFile, Header, Depends,Request
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates

from config.minio_config import minio_config
from config.settings import settings
from processor.query_processor.logger import logger
from utils.celery_tasks import kb_import_task
from utils.minio_utils import get_minio_client
from utils.task_utils import add_done_task, add_running_task, get_task_status, get_done_task_list, \
    get_running_task_list, get_task_result, get_task_create_time

# ========== 上传文件校验配置 ==========
ALLOWED_EXTENSIONS = {".pdf", ".md"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 单文件最大100MB
PDF_MAGIC_BYTES = b"%PDF-"  # PDF文件头魔数


def _validate_uploaded_file(file_path: str, filename: str) -> None:
    """
    校验上传文件：大小限制 + 真实文件类型（防改后缀绕过）
    校验不通过则删除文件并抛出HTTPException
    """
    # 1. 文件大小校验
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=f"文件过大：{file_size / 1024 / 1024:.1f}MB，单文件限制{MAX_FILE_SIZE // 1024 // 1024}MB"
        )

    # 2. 后缀名校验
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}，仅支持PDF/MD")

    # 3. 真实文件类型校验（读文件头魔数，防改后缀绕过）
    with open(file_path, "rb") as f:
        header = f.read(8)
    if suffix == ".pdf" and not header.startswith(PDF_MAGIC_BYTES):
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="文件后缀为.pdf但内容不是有效PDF文件")

# 1. 创建应用
# 标题和描述会在Swagger文档中展示
app = FastAPI(
    title="知识库问答-导入API",
    description="此文档是知识库问答导入流程的API接口说明"
)
async def verify_api_key(x_api_key: str = Header(None,alias ="X-API-Key")):
    """API 鉴权：未配置 API_KEY 时跳过（开发环境），配置后强制校验"""
    if settings.api_key and x_api_key !=settings.api_key:
        raise HTTPException(status_code=401,detail="无效的API Key")
    return x_api_key
# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins else ["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["GET","POST"],  # 允许的请求方法
    allow_headers=["X-API-Key","Content-Type"],  # 允许的请求头
)

# 3. 静态页面路由：服务端模板注入 API Key
# 访问地址：http://localhost:8001/import.html
templates = Jinja2Templates(directory=Path(__file__).absolute().parent.parent/ "page")

@app.get("/import.html") #对外访问地址
async def get_import_page(request:Request):
    # 拼接HTML文件绝对路径
    # current_dir_parent_path = Path(__file__).absolute().parent.parent
    # html_path = current_dir_parent_path / "page" / "import.html"

    # 如果不存在，抛出404异常
    # if not html_path.exists():
    #     raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    # return FileResponse(html_path)
    return templates.TemplateResponse("import.html",{
        "request":request,
        "api_key":settings.api_key,
    })


# 4. 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8001/upload （POST请求，form-data格式传参）
@app.post("/upload",summary="文件上传接口",description="支持多文件批量上传，自动触发知识库导入全流程",dependencies=[Depends(verify_api_key)])
async def upload_files(files:List[UploadFile] = File(...)):
    """
       文件上传核心接口
       1. 接收前端上传的多文件（PDF/MD为主）
       2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
       3. 将文件上传至MinIO对象存储，做持久化保存
       4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
       5. 实时更新任务状态，供前端轮询监控进度

       :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
       :param files: 前端上传的文件列表（form-data格式）
       :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 限制单次上传文件数量，防止批量上传打满磁盘
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="单次最多上传10个文件")
    # 1. 构建本地存储根目录：项目根目录/doc/YYYYMMDD（按日期分层，方便管理）
    data_based_root_dir = os.getenv("DATA_BASED_ROOT_DIR")
    if not data_based_root_dir:
        raise HTTPException(status_code=500, detail="环境变量 DATA_BASED_ROOT_DIR 未配置")
    if not os.path.exists(data_based_root_dir):
        os.makedirs(data_based_root_dir, exist_ok=True)

    data_str = datetime.datetime.now().strftime("%Y%m%d")
    data_dir = os.path.join(data_based_root_dir,data_str)
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id,"upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        file_dir = os.path.join(data_dir,task_id)
        os.makedirs(file_dir,exist_ok=True)
        # 构建上传文件的本地保存绝对路径
        import_file_path = os.path.join(file_dir,file.filename)


        # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
        with open(import_file_path,"wb") as file_buffer:
            #shutil 是 Python 标准库，专门做文件拷贝。
            shutil.copyfileobj(file.file,file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{import_file_path}")

        # 5.1 校验文件大小和真实类型（不通过会自动删文件并抛异常）
        _validate_uploaded_file(import_file_path, file.filename)

        # 6. 将本地文件上传至MinIO对象存储，做持久化保存
        # 构建MinIO中的文件对象名：pdf_files/YYYYMMDD/文件名（按日期分层，和本地一致）
        minio_object_name = f"pdf_files/{data_str}/{task_id}/{file.filename}"
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()

            # 从环境变量获取MinIO的桶名配置
            minio_bucket_name = minio_config.bucket_name

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=import_file_path,
                content_type=file.content_type
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            # MinIO上传失败，记录警告日志（不中断后续流程，本地文件仍可继续处理
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id,"upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        kb_import_task.delay(task_id,file_dir,import_file_path)
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

        # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
        logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code":200,
        "message":f"文件上传成功，total:{len(files)}",
        "task_ids":task_ids
    }

# 5. 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8001/status/{task_id} （GET请求）
@app.get("/status/{task_id}",summary="任务状态查询",description="根据TaskID查询单个文件的处理进度和全局状态",dependencies=[Depends(verify_api_key)])
async def get_task_progress(task_id:str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info : Dict[str,Any] = {
        "code":200,
        "task_id":task_id,
        "status":get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list":get_done_task_list(task_id), # 已完成的节点/阶段列表
        "running_list":get_running_task_list(task_id) ,# 正在运行的节点/阶段列表
        "progress": get_task_result(task_id, "node_progress", ""),
        "create_time": get_task_create_time(task_id)
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

if __name__ == "__main__":
    uvicorn.run(app=app,host="127.0.0.1",port=8001)