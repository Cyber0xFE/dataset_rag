import uuid
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import Field, BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from fastapi import Request

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
    set_task_result,
    get_task_result,
)
from app.utils.sse_utils import create_sse_queue, sse_generator, push_to_session
from app.query_process.agent.main_graph import kb_query_app
from app.core.logger import logger

_pending_queries: dict[str, str] = {}
_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="query service", description="掌柜智库查询服务！")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CHAT_HTML = PROJECT_ROOT / "app/query_process/page/chat.html"


@app.get("/chat", response_class=FileResponse)
async def chat_page():
    return FileResponse(_CHAT_HTML)


@app.get("/health")
async def health():
    return {"ok": True}


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")


def _run_query_pipeline(session_id: str, query: str, is_stream: bool = False) -> None:
    update_task_status(session_id, "processing")
    try:
        from app.query_process.agent.QueryGraphState import QueryGraphState

        state = QueryGraphState(
            session_id=session_id,
            original_query=query,
            is_stream=is_stream,
            embedding_chunks=[],
            hyde_embedding_chunks=[],
            kg_chunks=[],
            web_search_docs=[],
            rrf_chunks=[],
            reranked_docs=[],
            prompt="",
            answer="",
            item_names=[],
            rewritten_query="",
            history=[],
        )

        for event in kb_query_app.stream(state, stream_mode="updates"):
            for node_name in event:
                node_output = event[node_name]
                if isinstance(node_output, dict) and node_output.get("answer"):
                    set_task_result(session_id, "answer", node_output["answer"])

            if is_stream:
                push_to_session(session_id, "progress", {
                    "status": get_task_status(session_id),
                    "done_list": get_done_task_list(session_id),
                    "running_list": get_running_task_list(session_id),
                })

        answer = get_task_result(session_id, "answer")
        update_task_status(session_id, "completed")

        if is_stream:
            push_to_session(session_id, "final", {
                "status": "completed",
                "answer": answer,
                "done_list": get_done_task_list(session_id),
                "running_list": get_running_task_list(session_id),
                "image_urls": [],
            })

        logger.info(f"[{session_id}] 查询流程完成")
    except Exception as e:
        update_task_status(session_id, "failed")
        set_task_result(session_id, "error", str(e))
        if is_stream:
            push_to_session(session_id, "error", {"error": str(e)})
        logger.error(f"[{session_id}] 查询流程异常: {e}")


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    session_id = req.session_id
    if not session_id:
        session_id = "sess-" + uuid.uuid4().hex[:16]

    if req.is_stream:
        create_sse_queue(session_id)
        _pending_queries[session_id] = req.query
        return {"session_id": session_id}

    _run_query_pipeline(session_id, req.query)
    return {
        "session_id": session_id,
        "answer": get_task_result(session_id, "answer"),
        "error": get_task_result(session_id, "error"),
        "image_urls": [],
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    query = _pending_queries.pop(session_id, None)
    if query:
        _executor.submit(_run_query_pipeline, session_id, query, True)

    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
