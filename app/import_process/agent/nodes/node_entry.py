import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start，当前状态：{format_state(state)}")
    add_running_task(state.get('task_id'), fun_name)

    # 校验 local_file_path
    if not state.get('local_file_path'):
        logger.error(f"[{fun_name}] local_file_path 不存在")
        return state

    # 根据文件后缀判断类型，设置对应解析开关
    if state.get('local_file_path').endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = state['local_file_path']
    elif state.get('local_file_path').endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = state['local_file_path']
    else:
        logger.error(f"[{fun_name}] 不支持的文件类型")
        return state

    # 提取文件无后缀纯名称，作为全局业务标识
    state['file_title'] = state['local_file_path'].split('/')[-1].split('.')[0]

    logger.info(f"[{fun_name}] end，当前状态：{format_state(state)}")
    add_done_task(state.get('task_id'), fun_name)

    return state
