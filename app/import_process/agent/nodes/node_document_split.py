import os
import re
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'], fun_name)

    try:
        md_content = state.get('md_content', '')
        md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
        if not md_content.strip():
            raise ValueError(f"[{fun_name}] md_content为空，无法切分")
        file_title = state.get('file_title', '')

        chunks = []
        in_code = False
        section_start = 0
        current_title = ""

        for m in re.finditer(r'(^```.*$)|(^(#+)\s+(.+)$)', md_content, re.MULTILINE):
            if m.group(1):
                in_code = not in_code
            elif not in_code and m.group(2):
                start = m.start()
                if section_start == 0 and start > 0:
                    chunks.append({
                        "title": "",
                        "content": md_content[:start].strip(),
                        "file_title": file_title,
                    })
                elif section_start > 0:
                    chunks.append({
                        "title": current_title,
                        "content": md_content[section_start:start].strip(),
                        "file_title": file_title,
                    })
                current_title = f"{m.group(3)} {m.group(4)}"
                section_start = start

        if section_start > 0:
            chunks.append({
                "title": current_title,
                "content": md_content[section_start:].strip(),
                "file_title": file_title,
            })

        # state['chunks'] = chunks
        logger.info(f"[{fun_name}] 切分为 {len(chunks)} 个块")

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'],fun_name)

    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\万用表RS-12的使用", "full_test.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": open(test_md_path, "r", encoding="utf-8").read(),
            "file_title": "万用表RS-12的使用",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        # logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        # logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")