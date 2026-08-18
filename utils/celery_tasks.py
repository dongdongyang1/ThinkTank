import logging

from config.celery_config import celery
from processor.import_processor.main_graph import KBImportWorkflow
from utils.task_utils import update_task_status, add_done_task


@celery.task(bind=True, max_retries=2) # 失败自动重试2次
def kb_import_task(self,task_id:str,file_dir:str,import_file_path:str):
    """Celery 知识库导入异步任务"""
    try:
        update_task_status(task_id,"processiong")
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }
        workflow = KBImportWorkflow()
        # 流式执行图节点
        for event in workflow.run(init_state,stream=True):
            for node_name,node_result in event.items():
                add_done_task(task_id,node_name)
                self.update_state(state="PROGRESS", meta={"current_node": node_name})

        # 全部节点跑完
        update_task_status(task_id,"completed")
        logging.getLogger().info(f"[{task_id}] 文档解析全流程执行完成")
        return {"task_id": task_id, "status": "completed"}
    except Exception as e:
        update_task_status(task_id, "failed")
        logging.getLogger().info(f"[{task_id}] LangGraph执行失败，异常：{str(e)}", exc_info=True)
        # 瞬时异常自动重试
        self.retry(exc=e, countdown=3)