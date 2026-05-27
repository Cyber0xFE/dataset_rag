import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

_BATCH_SIZE = 5


def _build_embeddings(chunks: list, item_name: str, logger_prefix: str) -> list:
    """批量生成向量，将 dense/sparse 向量合并回每个 chunk。"""
    result = []
    total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start:batch_start + _BATCH_SIZE]
        texts = [f"商品：{item_name}，介绍：{chunk['content']}" for chunk in batch]
        embeddings = generate_embeddings(texts)

        for i, chunk in enumerate(batch):
            result.append({
                **chunk,
                "item_name": item_name,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": embeddings["sparse"][i],
            })

        logger.info(f"[{logger_prefix}] 已处理批次 {batch_start // _BATCH_SIZE + 1}/{total_batches} ({len(batch)} 条)")

    return result


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'], fun_name)

    try:
        item_name = state.get('item_name', '')
        chunks = state.get('chunks', [])
        if not chunks:
            raise ValueError(f"[{fun_name}] chunks 为空，无法生成向量")

        state['chunks'] = _build_embeddings(chunks, item_name, fun_name)

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'], fun_name)

    return state


# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
def test_node_bge_embedding():
    """本地测试：加载真实 chunks JSON 数据，独立验证向量化节点全链路。"""
    import json
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.state import create_default_state

    logger.info("=== BGE-M3 向量化节点本地单元测试启动 ===")

    chunks_path = PROJECT_ROOT / "output" / "万用表RS-12的使用_chunks.json"
    if not chunks_path.exists():
        logger.error(f"测试文件不存在: {chunks_path}")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    test_state = create_default_state(
        task_id="test_task_embedding_001",
        file_title=chunks[0]["file_title"],
        chunks=chunks,
    )

    result_state = node_bge_embedding(test_state)
    result_chunks = result_state.get("chunks", [])

    logger.info("=== 向量化节点本地测试完成 ===")
    logger.info(f"任务ID: {test_state.get('task_id')}")
    logger.info(f"待处理切片数: {len(chunks)} | 实际处理: {len(result_chunks)}")

    for idx, chunk in enumerate(result_chunks):
        has_dense = "dense_vector" in chunk
        has_sparse = "sparse_vector" in chunk
        logger.info(f"第{idx + 1}条: dense={has_dense}, sparse={has_sparse}")


if __name__ == '__main__':
    test_node_bge_embedding()