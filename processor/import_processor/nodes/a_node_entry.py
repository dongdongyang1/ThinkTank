import json
import logging
from pathlib import Path

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError,ValidationError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务开发
    """
    name = "a_node_entry"  #属性
    def process(self, state: ImportGraphState) :  #方法
        """
        1.接收状态：获取import_file_path
        2.判断类型：检查文件后缀是.pdf还是.md
        3.设置标记：更新state中is_pdf_read_enabled/pdf_path或is_md_read_enabled/md_path,供主图路由使用
        4.提取标题：从文件名中提取file_title,后续作为元数据
        :param state: ‘import_file_path'
        :return: 'is_pdf_read_enabled/pdf_path'或'is_md_read_enabled/md_path
        """

        # 1. 从state中获取文件
        ## import_file_path = state["import_file_path"] #这样写的话字段不存在直接报错
        import_file_path = state.get("import_file_path")

        # 判断路径是否为空
        if not import_file_path:
            raise StateFieldError(filed_name="import_file_path",expected_type=str)

        # 2. 转换path标准化对象
        import_file_path_obj = Path(import_file_path)

        # 判断文件是否存在
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件{import_file_path_obj.name}不存在")




        # 3. 检查文件后缀
        file_suffix = import_file_path_obj.suffix.lower()
        if file_suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path

        elif file_suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
            md_path_obj = Path(state["md_path"])
            # 读取MD文件内容存入state
            md_content = md_path_obj.read_text(encoding="utf-8")
            state["md_content"] = md_content
        else:
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.stem}不支持")

        # 4. 获取上传文件的标题，更新到state中
        state["file_title"] = import_file_path_obj.stem
        state["file_dir"] = str(import_file_path_obj.parent)

        # 5. 返回state
        return state


if __name__ == "__main__":

    setup_logging()

    init_state = {"import_file_path":r"D:\hak180产品安全手册.pdf"}

    node_entry = NodeEntry()
    result = node_entry(init_state)

    logging.getLogger().info(json.dumps(result,ensure_ascii=False,indent=4))



