"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging
import logging.handlers
import os
from pathlib import Path

from processor.import_processor.import_config import ImportConfig, get_config
from processor.import_processor.exceptions import ImportProcessError
from utils.task_utils import add_running_task, add_done_task, remove_running_task

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")
            add_running_task(state["task_id"],self.name)

            # 2. 执行节点
            result = self.process(state)

            # 3. 执行节点成功
            add_done_task(state["task_id"],self.name)
            self.logger.info(f"--- {self.name} 完成 ---")

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}",exc_info=True)
            # 失败时从"运行中节点"列表移除，避免前端在failed状态下仍显示该节点运行中
            task_id = state.get("task_id")
            if task_id:
                remove_running_task(task_id, self.name)
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志：控制台输出 + 滚动文件持久化

    Args:
        level: 日志级别
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 避免重复添加handler（多次调用setup_logging时）
    if root_logger.handlers:
        return

    # 1. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. 文件持久化（按大小滚动，单文件10MB，保留5个备份）
    log_dir = os.getenv("LOG_DIR", str(Path(__file__).resolve().parent.parent.parent / "logs"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "import_processor.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)





