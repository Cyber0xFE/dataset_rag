import os
import sys
from pathlib import Path

from pymilvus import DataType, MilvusClient

from app.clients.milvus_utils import get_milvus_client
from app.conf.lm_config import lm_config
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm import lm_utils
from app.lm.embedding_utils import generate_embeddings

# 商品名称识别上下文常量
ITEM_NAME_CHUNK_K = 5
ITEM_NAME_MAX_CHARS = 2500
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task


def _build_recognition_context(chunks: list, k: int, max_chars: int) -> str:
    """从 chunks 中提取前 k 个切片，按指定格式构建上下文字符串。"""
    parts = []
    total = 0
    for idx, chunk in enumerate(chunks[:k]):
        title = chunk.get("title", "").strip()
        content = chunk.get("content", "").strip()
        if not (title or content):
            continue
        piece = f"切片：{idx + 1}，标题：{title}，内容：{content}"
        total += len(piece)
        if total > max_chars:
            break
        parts.append(piece)
    return "\n".join(parts)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用大模型，返回响应文本。"""
    llm = lm_utils.get_llm_client(lm_config.llm_model)
    r = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return r.content.strip()


def _escape_milvus_string(value: str) -> str:
    """Milvus filter 表达式字符串转义。"""
    return escape_milvus_string(value)


def _save_item_name_to_milvus(client: MilvusClient, collection_name: str, file_title: str, item_name: str, dense_vec: list, sparse_vec: dict) -> None:
    """将 item_name 及其向量写入 Milvus（集合不存在则自动创建）。"""
    if not client.has_collection(collection_name):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("file_title", DataType.VARCHAR, max_length=512)
        schema.add_field("item_name", DataType.VARCHAR, max_length=512)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index("dense_vector", index_name="idx_dense_vector",
                               index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        index_params.add_index("sparse_vector", index_name="idx_sparse_vector",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                               params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"})

        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        logger.info(f"已创建 Milvus 集合: {collection_name}")

    safe_name = _escape_milvus_string(item_name)
    client.delete(collection_name, filter=f'item_name == "{safe_name}"')
    client.insert(collection_name, [{
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vec,
        "sparse_vector": sparse_vec,
    }])
    client.load_collection(collection_name=collection_name)
    logger.info(f"已插入 item_name 向量至 {collection_name}")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    fun_name = sys._getframe().f_code.co_name

    logger.info(f"[{fun_name}] start")
    add_running_task(state['task_id'], fun_name)

    try:
        if not state.get('chunks'):
            raise ValueError(f"[{fun_name}] chunks 为空，无法继续处理")
        if not state.get('file_title'):
            state['file_title'] = Path(state.get('md_path', '')).stem

        context = _build_recognition_context(state['chunks'], ITEM_NAME_CHUNK_K, ITEM_NAME_MAX_CHARS)
        user_prompt = load_prompt('item_name_recognition', file_title=state['file_title'], context=context)
        system_prompt = load_prompt('product_recognition_system')
        state['item_name'] = _call_llm(system_prompt, user_prompt)
        logger.info(f"[{fun_name}] 识别结果: {state['item_name']}")

        embeddings = generate_embeddings([state['item_name']])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        client = get_milvus_client()
        if not client:
            raise ConnectionError(f"[{fun_name}] 无法连接 Milvus")
        _save_item_name_to_milvus(client, milvus_config.item_name_collection, state['file_title'], state['item_name'], dense_vec, sparse_vec)

    except Exception as e:
        logger.error(f"[{fun_name}] error: {e}")
        raise e
    finally:
        logger.info(f"[{fun_name}] end")
        add_done_task(state['task_id'],fun_name)

    return state


# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = _escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()