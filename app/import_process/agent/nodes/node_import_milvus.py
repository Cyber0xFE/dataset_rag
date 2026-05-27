import sys

from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task


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


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'], fun_name)

    try:
        item_name = state.get('item_name', '')
        chunks = state.get('chunks', [])
        if not chunks:
            raise ValueError(f"[{fun_name}] chunks 为空，无法导入 Milvus")

        client = get_milvus_client()
        if not client:
            raise ConnectionError(f"[{fun_name}] 无法连接 Milvus")
        _ensure_chunks_collection(client)

        safe_name = escape_milvus_string(item_name)
        delete_result = client.delete(milvus_config.chunks_collection, filter=f'item_name == "{safe_name}"')
        logger.info(f"[{fun_name}] 已清理 item_name={item_name} 的旧数据，删除 {delete_result.get('delete_count', 0)} 条")

        insert_result = client.insert(milvus_config.chunks_collection, chunks)
        auto_ids = insert_result.get("ids", [])
        for chunk, chunk_id in zip(chunks, auto_ids):
            chunk['chunk_id'] = chunk_id
        logger.info(f"[{fun_name}] 已插入 {len(auto_ids)} 条数据")

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'], fun_name)

    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name": "万用表RS-12",
        "chunks": [
            {
                "content": "产品概述：万用表RS-12是一款高精度数字万用表。",
                "title": "产品概述",
                "item_name": "万用表RS-12",
                "parent_title": "test.pdf",
                "part": 0,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            },
            {
                "content": "技术参数：直流电压量程0-600V，交流电压量程0-600V。",
                "title": "技术参数",
                "item_name": "万用表RS-12",
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,
                "sparse_vector": {2: 0.6, 11: 0.7},
            },
            {
                "content": "使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。",
                "title": "使用方法",
                "item_name": "万用表RS-12",
                "parent_title": "test.pdf",
                "part": 2,
                "file_title": "test.pdf",
                "dense_vector": [0.3] * dim,
                "sparse_vector": {3: 0.7, 12: 0.6},
            },
            {
                "content": "安全警告：测量电压时请勿超过额定输入值，防止触电。",
                "title": "安全警告",
                "item_name": "万用表RS-12",
                "parent_title": "test.pdf",
                "part": 3,
                "file_title": "test.pdf",
                "dense_vector": [0.4] * dim,
                "sparse_vector": {4: 0.8, 13: 0.5},
            },
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")