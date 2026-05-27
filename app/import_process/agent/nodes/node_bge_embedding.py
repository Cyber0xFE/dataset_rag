import sys

from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

_BATCH_SIZE = 5


def _ensure_chunks_collection(client: MilvusClient) -> None:
    """确保 Milvus 中 chunks 集合已创建（不存在则自动创建）。"""
    collection_name = milvus_config.chunks_collection
    if client.has_collection(collection_name):
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("file_title", DataType.VARCHAR, max_length=512)
    schema.add_field("item_name", DataType.VARCHAR, max_length=512)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("parent_title", DataType.VARCHAR, max_length=512)
    schema.add_field("part", DataType.INT8)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("dense_vector", index_name="idx_dense_vector",
                           index_type="HNSW", metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index("sparse_vector", index_name="idx_sparse_vector",
                           index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                           params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"})

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"已创建 Milvus 集合: {collection_name}")


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

        client = get_milvus_client()
        if not client:
            raise ConnectionError(f"[{fun_name}] 无法连接 Milvus")
        _ensure_chunks_collection(client)

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'], fun_name)

    return state
