import json
import os
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger

# 递归切割器常量
MAX_CHUNK_SIZE = 100
CHUNK_OVERLAP = 10
MIN_CHUNK_SIZE = 50
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def _split_by_headings(md_content: str, file_title: str) -> list:
    """按标题切分 markdown，跳过代码块内的 #，返回 chunk 列表。"""
    chunks = []
    in_code = False
    section_start = 0
    current_title = ""

    for m in re.finditer(r'(^(?:```|~~~).*$)|(^(#+)\s+(.+)$)', md_content, re.MULTILINE):
        if m.group(1):
            in_code = not in_code
        elif not in_code and m.group(2):
            start = m.start()
            if section_start == 0 and start > 0:
                chunks.append({
                    "title": "default_title_preamble",
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
    elif not chunks:
        chunks.append({
            "title": "default_title_document",
            "content": md_content.strip(),
            "file_title": file_title,
        })

    return chunks


def _ensure_overlap(texts: list, overlap: int) -> list:
    """保证相邻文本块之间有 overlap 字符的重叠（弥补 RecursiveCharacterTextSplitter 跨层级不重叠的问题）。"""
    if len(texts) <= 1 or overlap <= 0:
        return texts
    result = [texts[0]]
    for i in range(1, len(texts)):
        prev = texts[i - 1]
        curr = texts[i]
        tail = prev[-overlap:] if len(prev) >= overlap else prev
        if tail and not curr.startswith(tail):
            curr = tail + curr
        result.append(curr)
    return result


def _merge_small_chunks(chunks: list, min_size: int) -> list:
    """合并小于 min_size 的 chunk 到前一个 chunk，重叠部分去重。"""
    if not chunks:
        return chunks
    result = [dict(chunks[0])]
    for i in range(1, len(chunks)):
        prev = result[-1]
        curr = chunks[i]
        if len(prev['content']) >= min_size or prev['parent_title'] != curr['parent_title']:
            result.append(dict(curr))
        else:
            # 合并到前一个 chunk，去除 prev 尾部与 curr 头部的重复
            prev_tail = prev['content']
            curr_head = curr['content']
            overlap = 0
            max_check = min(len(prev_tail), len(curr_head))
            for ol in range(max_check, 0, -1):
                if prev_tail[-ol:] == curr_head[:ol]:
                    overlap = ol
                    break
            prev['content'] = prev_tail + curr_head[overlap:]
    return result


def _protect_images(text: str) -> tuple[str, list[str]]:
    """保护 Markdown 图片链接，替换为占位符，返回 (替换后文本, 图片列表)。"""
    images = re.findall(r'!\[.*?\]\(.*?\)', text)
    for i, img in enumerate(images):
        text = text.replace(img, f"__IMG_{i}__", 1)
    return text, images


def _restore_images(text: str, images: list[str]) -> str:
    """将占位符还原为原始图片链接。"""
    for i, img in enumerate(images):
        text = text.replace(f"__IMG_{i}__", img)
    return text


def _split_chunks(heading_chunks: list) -> list:
    """用 RecursiveCharacterTextSplitter 对每个 heading chunk 递归切割，补 overlap。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", "。", "！", "？", "．", "，", ",", " ", ""]
    )
    chunks = []
    for c in heading_chunks:
        content, images = _protect_images(c['content'])
        sub_texts = splitter.split_text(content)
        sub_texts = _ensure_overlap(sub_texts, CHUNK_OVERLAP)
        sub_texts = [_restore_images(st, images) for st in sub_texts]
        for part_no, st in enumerate(sub_texts):
            chunks.append({
                "title": f"{c['title']}_{part_no}",
                "content": st,
                "file_title": c['file_title'],
                "parent_title": c['title'],
                "part": part_no,
            })
    return chunks


def _save_chunks_json(chunks: list, file_title: str, output_dir: str) -> None:
    """将 chunks 序列化为 JSON 保存到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{file_title}_chunks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"chunks 已保存至 {json_path}")


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

        heading_chunks = _split_by_headings(md_content, file_title)

        chunks = _split_chunks(heading_chunks)
        # 合并过小的 chunk，保证最小大小
        chunks = _merge_small_chunks(chunks, MIN_CHUNK_SIZE)

        state['chunks'] = chunks
        _save_chunks_json(chunks, file_title, state.get('local_dir', os.path.dirname(state.get('md_path', ''))))
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

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\万用表RS-12的使用", "full_new.md")
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

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")