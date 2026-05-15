import os
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'],fun_name)

    try:
        # 校验pdf_path如果不存在则抛异常
        pdf_path = state.get('pdf_path', '')
        if not pdf_path:
            raise ValueError(f"[{fun_name}] state中缺少pdf_path字段")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"[{fun_name}] PDF文件不存在: {pdf_path}")

        if not state.get('local_dir'):
            from pathlib import Path
            local_dir = PROJECT_ROOT / "output" / Path(pdf_path).stem
            state['local_dir'] = str(local_dir)
            logger.info(f"[{fun_name}] 默认local_dir: {local_dir}")

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'],fun_name)

    return state