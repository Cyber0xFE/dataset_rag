import base64
import os
import re
import sys
from pathlib import Path

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm import lm_utils
from app.utils.task_utils import add_running_task


def _batch_upload_to_minio(md_dir: str, img_paths: list) -> dict:
    """批量上传本地图片到MinIO，返回 {img_path: url} 映射。"""
    minio_client = get_minio_client()
    protocol = "https" if minio_config.minio_secure else "http"
    url_map = {}
    for img_path in img_paths:
        img_full_path = os.path.join(md_dir, img_path)
        object_name = f"{minio_config.minio_img_dir}/{img_path}"
        minio_client.fput_object(minio_config.bucket_name, object_name, img_full_path)
        url_map[img_path] = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
    return url_map


def _image_summary(lv, md_dir: str, root_folder: str, img_path: str, before: str, after: str) -> str:
    """加载提示词，将本地图片 base64 编码后调用 VL 模型，返回图片摘要。"""
    img_full_path = os.path.join(md_dir, img_path)
    with open(img_full_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    img_suffix = os.path.splitext(img_path)[1].lstrip(".") or "png"
    img_data_uri = f"data:image/{img_suffix};base64,{img_b64}"

    prompt = load_prompt('image_summary', root_folder=root_folder, image_content=[before, after])
    r = lv.invoke([
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_uri},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ])
    return r.content


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'], fun_name)

    # 读取md文件内容
    with open(state['md_path'], "r", encoding="utf-8") as f:
        md_content = f.read()

    # 正则匹配图片并提取前后100字符上下文，一步到位
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+?)\)')
    img_contexts = []
    for m in img_pattern.finditer(md_content):
        alt_text = m.group(1)
        img_path = m.group(2)
        start, end = m.start(), m.end()
        before = md_content[max(0, start - 100):start]
        after = md_content[end:end + 100]
        img_contexts.append((alt_text, img_path, [before, after]))
    logger.info(f"[{fun_name}] 提取到 {len(img_contexts)} 张图片: {[img for _, img, _ in img_contexts]}")

    lv = lm_utils.get_llm_client(lm_config.lv_model)

    md_dir = os.path.dirname(state['md_path'])
    root_folder = Path(state['md_path']).stem

    # 清空MinIO bucket中的旧文件
    minio_client = get_minio_client()
    objects = list(minio_client.list_objects(minio_config.bucket_name))
    if objects:
        minio_client.remove_objects(minio_config.bucket_name, [o.object_name for o in objects])
        logger.info(f"[{fun_name}] 已清空bucket({len(objects)}个文件)")
    # 批量上传图片到MinIO
    img_url_map = _batch_upload_to_minio(md_dir, [img_path for _, img_path, _ in img_contexts])
    logger.info(f"[{fun_name}] 批量上传完成：{img_url_map}")

    for img_context in img_contexts:
        alt_text, img_path, (before, after) = img_context
        summary = _image_summary(lv, md_dir, root_folder, img_path, before, after)
        logger.info(f"[{fun_name}] 图片摘要：{summary}")

        img_url = img_url_map[img_path]
        old_ref = f'![{alt_text}]({img_path})'
        new_ref = f'![{summary}]({img_url})'
        md_content = md_content.replace(old_ref, new_ref, 1)

    state['md_content'] = md_content
    md_path = Path(state['md_path'])
    new_md_path = md_path.with_name(f"{md_path.stem}_new{md_path.suffix}")
    new_md_path.write_text(md_content, encoding="utf-8")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\万用表RS-12的使用", "full.md")
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")    