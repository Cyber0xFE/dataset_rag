import sys

from app.core.logger import logger
from app.query_process.agent.QueryGraphState import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.milvus_utils import get_milvus_client, hybrid_search, create_hybrid_search_requests
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.utils.escape_milvus_string_utils import escape_milvus_string


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query", state.get("original_query", ""))

    client = get_milvus_client()
    client.load_collection(milvus_config.chunks_collection)

    try:
        # 生成改写后问题的向量
        embeddings = generate_embeddings([rewritten_query])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        # 构建 item_name 过滤条件
        safe_names = ", ".join(f'"{escape_milvus_string(name)}"' for name in item_names)
        expr = f"item_name in [{safe_names}]"
        logger.info(f"向量检索过滤条件: {expr}")

        # 混合搜索
        reqs = create_hybrid_search_requests(dense_vec, sparse_vec, expr=expr, limit=10)
        res = hybrid_search(
            client, milvus_config.chunks_collection, reqs,
            ranker_weights=(0.8, 0.2), norm_score=True, limit=5,
            output_fields=["chunk_id", "content", "title", "parent_title", "item_name", "file_title"],
        )

        chunks = []
        if res and res[0]:
            for hit in res[0]:
                chunks.append(hit["entity"])
            logger.info(f"向量检索到 {len(chunks)} 个切片")
        else:
            logger.warning("向量检索无结果")

        state["embedding_chunks"] = chunks

    except Exception as e:
        logger.error(f"向量检索异常: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "万用表RS-12使用说明",  # 模拟改写后的查询
        "item_names": ["万用表RS-12"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)