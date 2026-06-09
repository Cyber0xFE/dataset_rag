import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/import", response_class=FileResponse)
async def get_import_page():
    """返回文件导入前端页面：import.html"""
    # 拼接HTML文件绝对路径，基于项目根目录定位
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    # 日志记录页面访问的文件路径，方便排查文件不存在问题
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")

    # 校验文件是否存在，不存在则抛出404异常
    if not os.path.exists(html_abs_path):
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")

    # 以FileResponse返回HTML文件，浏览器自动渲染
    return FileResponse(
        path=html_abs_path,
        media_type="text/html"  # 显式指定媒体类型为HTML，确保浏览器正确解析
    )

def _run_import_pipeline(task_id: str, task_dir: str) -> None:
    """后台执行 LangGraph 导入全流程。"""
    update_task_status(task_id, "processing")

    local_files = [f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))]
    if not local_files:
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] 任务目录无文件")
        return

    try:
        state = get_default_state()
        state["task_id"] = task_id
        state["local_dir"] = task_dir
        state["local_file_path"] = os.path.join(task_dir, local_files[0])

        for event in kb_import_app.stream(state, stream_mode="updates"):
            for node_name in event:
                logger.info(f"[{task_id}] 节点完成: {node_name}")

        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] 导入流程完成")
    except Exception as e:
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] 导入流程异常: {e}")


@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """文件上传接口：保存文件并触发 LangGraph 导入流程。"""
    base_dir = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d")
    os.makedirs(base_dir, exist_ok=True)

    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        add_running_task(task_id, "upload_file")

        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        file_path = os.path.join(task_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        add_done_task(task_id, "upload_file")

        background_tasks.add_task(_run_import_pipeline, task_id, task_dir)

    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids,
    }

@app.get("/status/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """查询指定任务的状态。"""
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }

# --------------------------
# 服务启动入口
# 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8002  # 服务端口
    )
