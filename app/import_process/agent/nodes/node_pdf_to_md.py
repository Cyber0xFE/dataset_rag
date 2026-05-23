import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task

# ── 常量 ──────────────────────────────────────────────────────────────

_POLL_TIMEOUT = 300          # 轮询超时（秒）
_POLL_INITIAL_INTERVAL = 1   # 轮询起始间隔（秒）
_POLL_MAX_INTERVAL = 30      # 轮询最大间隔（秒）
_REQUEST_RETRIES = 3         # 请求重试次数
_MODEL_VERSION = "vlm"       # MinerU 模型版本

# ── 请求会话 & 公共 header（复用连接，避免重复构造） ──────────────────

_session = requests.Session()
_HEADER = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {mineru_config.api_key}",
}


def _request_retry(method: str, url: str, **kwargs) -> requests.Response:
    """带退避重试的 HTTP 请求，仅对可重试错误（网络异常/5xx）重试。"""
    last_exc = None
    for attempt in range(1, _REQUEST_RETRIES + 1):
        try:
            resp = _session.request(method, url, timeout=120, **kwargs)
            if resp.status_code < 500:
                return resp
            logger.warning(f"服务端 {resp.status_code}，重试 {attempt}/{_REQUEST_RETRIES}")
        except requests.RequestException as e:
            logger.warning(f"请求异常: {e}，重试 {attempt}/{_REQUEST_RETRIES}")
            last_exc = e
        if attempt < _REQUEST_RETRIES:
            time.sleep(attempt * 2)  # 退避：2, 4, 6 秒
    raise last_exc or RuntimeError(f"请求失败，已达最大重试次数({_REQUEST_RETRIES})")


def mineru_extract(pdf_path: str) -> str:
    """调用 MinerU API 上传 PDF 并轮询提取结果。

    Args:
        pdf_path: 本地 PDF 文件路径。

    Returns:
        提取结果 zip 文件的下载地址 (full_zip_url)。
    """
    # 1. 获取文件上传 URL
    url = f"{mineru_config.base_url}/file-urls/batch"
    data = {
        "files": [
            {"name": pdf_path, "data_id": Path(pdf_path).stem}
        ],
        "model_version": _MODEL_VERSION,
    }

    resp = _request_retry("POST", url, headers=_HEADER, json=data)
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"获取文件上传URL失败: {result.get('msg')}")

    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]
    logger.info(f"batch_id:{batch_id}, urls:{urls}", batch_id)

    # 2. 上传文件
    with open(pdf_path, "rb") as f:
        resp_up = _request_retry("PUT", urls[0], data=f)
        if resp_up.status_code != 200:
            raise RuntimeError(f"文件上传失败: {resp_up.status_code}")
        logger.info(f"文件上传成功: {urls[0]}")

    # 3. 轮询提取结果（指数退避）
    start_time = time.time()
    poll_interval = _POLL_INITIAL_INTERVAL
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    while True:
        elapsed = time.time() - start_time
        if elapsed > _POLL_TIMEOUT:
            raise TimeoutError(f"MinerU 提取超时({_POLL_TIMEOUT}s)，batch_id: {batch_id}")

        try:
            resp = _session.get(poll_url, headers=_HEADER, timeout=30)
            result = resp.json()
        except requests.RequestException:
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, _POLL_MAX_INTERVAL)
            continue

        if resp.status_code == 200 and result.get("code") == 0:
            extract_result = result["data"]["extract_result"]
            if not extract_result:
                time.sleep(poll_interval)
                continue

            state = extract_result[0].get("state")
            if state == "done":
                return extract_result[0]["full_zip_url"]
            elif state in ("failed", "error"):
                raise RuntimeError(
                    f"MinerU 提取失败，batch_id: {batch_id}, state: {state}, "
                    f"msg: {extract_result[0].get('msg', '')}"
                )

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 2, _POLL_MAX_INTERVAL)


def _download_and_extract(pdf_path: str, full_zip_url: str) -> str:
    """下载 MinerU 提取结果 zip 并解压，返回其中的 .md 文件路径。

    Args:
        pdf_path: PDF 文件路径（用于确定输出目录名）。
        full_zip_url: MinerU 返回的 zip 下载地址。

    Returns:
        解压目录中找到的第一个 .md 文件路径。

    Raises:
        FileNotFoundError: 解压目录中未找到 .md 文件。
    """
    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_path).stem
    zip_path = out_dir / f"{stem}.zip"

    logger.info(f"开始下载: {full_zip_url}")
    with _session.get(full_zip_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info(f"下载完成: {zip_path}")

    extract_dir = out_dir / stem
    if extract_dir.exists():
        logger.info(f"目录已存在，删除: {extract_dir}")
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info(f"解压完成: {extract_dir}")

    md_files = list(extract_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"解压目录中未找到 .md 文件: {extract_dir}")
    logger.info(f"发现 {len(md_files)} 个 .md 文件: {[f.name for f in md_files]}")

    return str(md_files[0])


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'],fun_name)

    try:
        pdf_path = state.get('pdf_path', '')
        if not pdf_path:
            raise ValueError(f"[{fun_name}] state中缺少pdf_path字段")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"[{fun_name}] PDF文件不存在: {pdf_path}")

        if not state.get('local_dir'):
            local_dir = PROJECT_ROOT / "output" / Path(pdf_path).stem
            state['local_dir'] = str(local_dir)
            logger.info(f"[{fun_name}] 默认local_dir: {local_dir}")

        full_zip_url = mineru_extract(pdf_path)
        md_path = _download_and_extract(pdf_path, full_zip_url)

        state['md_path'] = md_path

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'],fun_name)

    return state