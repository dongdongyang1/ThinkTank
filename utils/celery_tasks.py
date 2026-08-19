import logging

from config.celery_config import celery
from processor.import_processor.exceptions import ImportProcessError
from processor.import_processor.main_graph import KBImportWorkflow
from utils.task_utils import update_task_status, add_done_task, set_task_result, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED


@celery.task(bind=True, max_retries=2, retry_backoff=True, retry_backoff_max=60, retry_jitter=True)
def kb_import_task(self, task_id: str, file_dir: str, import_file_path: str):
    """Celery 知识库导入异步任务"""
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }
        workflow = KBImportWorkflow()
        # 流式执行图节点，累积最终状态
        final_state = init_state.copy()
        for event in workflow.run(init_state, stream=True):
            for node_name, node_result in event.items():
                add_done_task(task_id, node_name)
                final_state.update(node_result)
                self.update_state(state="PROGRESS", meta={"current_node": node_name})

        # 检查业务层面的失败（如PDF解析超限，节点内部吞了异常不会raise）
        if final_state.get("pdf_parse_error"):
            err_msg = final_state["pdf_parse_error"]
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error_msg", err_msg)
            logging.getLogger().warning(f"[{task_id}] 导入任务业务失败：{err_msg}")
            return {"task_id": task_id, "status": "failed", "error": err_msg}

        # 全部节点跑完
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logging.getLogger().info(f"[{task_id}] 文档解析全流程执行完成")
        return {"task_id": task_id, "status": "completed"}
    except ImportProcessError as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        set_task_result(task_id, "error_msg", str(e))
        raise
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        set_task_result(task_id, "error_msg", str(e))
        # 原代码 self.logger() 是bug：celery 5 中 self.logger 不是方法，这段分支从未跑过所以没炸
        logging.getLogger(__name__).info(f"[{task_id}] LangGraph执行失败，异常：{str(e)}", exc_info=True)
        # 瞬时异常自动重试。注意：会整条流水线重跑，MinerU 会重复上传解析，如不能接受删掉这行并把max_retries=0
        self.retry(exc=e, countdown=3)
