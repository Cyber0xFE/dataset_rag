import sys

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.task_utils import add_done_task, add_running_task
from app.clients.milvus_utils import get_milvus_client, hybrid_search, create_hybrid_search_requests
from app.conf.milvus_config import milvus_config
from app.conf.lm_config import lm_config
from app.lm import lm_utils
from app.lm.embedding_utils import generate_embeddings
from app.utils.escape_milvus_string_utils import escape_milvus_string


def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query", state.get("original_query", ""))
    item_names = state.get("item_names", [])

    # 1. 加载提示词，请求 LLM 生成假设性答案
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    llm = lm_utils.get_llm_client(lm_config.llm_model)
    resp = llm.invoke([{"role": "user", "content": prompt}])
    hyde_answer = resp.content.strip()
    logger.info(f"HyDE 生成假设性答案: {hyde_answer[:100]}...")

    # 2. 依据假设性答案与 milvus 做向量检索
    client = get_milvus_client()
    client.load_collection(milvus_config.chunks_collection)

    try:
        embeddings = generate_embeddings([hyde_answer])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        safe_names = ", ".join(f'"{escape_milvus_string(name)}"' for name in item_names)
        expr = f"item_name in [{safe_names}]"

        reqs = create_hybrid_search_requests(dense_vec, sparse_vec, expr=expr, limit=10)
        res = hybrid_search(
            client, milvus_config.chunks_collection, reqs,
            ranker_weights=(0.8, 0.2), norm_score=True, limit=10,
            output_fields=["chunk_id", "content", "title", "parent_title", "item_name", "file_title"],
        )

        chunks = []
        if res and res[0]:
            for hit in res[0]:
                chunks.append(hit["entity"])
            logger.info(f"HyDE 向量检索到 {len(chunks)} 个切片")
        else:
            logger.warning("HyDE 向量检索无结果")

        state["hyde_embedding_chunks"] = chunks

    except Exception as e:
        logger.error(f"HyDE 检索异常: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": state.get("hyde_embedding_chunks", [])}


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "万用表RS-12怎么操作？",
        "rewritten_query": "万用表RS-12的具体操作步骤是什么？",
        "item_names": ["万用表RS-12"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")