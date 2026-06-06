import sys

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task


# RRF 常数
_RRF_K = 60
_RRF_TOP_K = 10


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """计算 RRF 分数。"""
    return 1.0 / (k + rank)


def node_rrf(state):
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 获取两路向量召回结果
    lists = {
        "embedding": state.get("embedding_chunks", []) or [],
        "hyde": state.get("hyde_embedding_chunks", []) or [],
    }

    # 每路召回权重（依次对应 lists 顺序）
    weights = (1.0, 1.0)

    # 对每路结果做 RRF 评分
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for idx, (name, lst) in enumerate(lists.items()):
        if not lst:
            logger.info(f"RRF 跳过空列表: {name}")
            continue
        w = weights[idx]
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + w * _rrf_score(rank)
            if cid not in items:
                items[cid] = item

    # 按 RRF 分数倒序排序，取 top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:_RRF_TOP_K]
    result = []
    for key, score in ranked:
        item = items[key]
        item["rrf_score"] = round(score, 4)
        result.append(item)

    state["rrf_chunks"] = result
    logger.info(f"RRF 融合完成，输入: {', '.join(f'{n}={len(l)}' for n, l in lists.items())}，输出: {len(result)} 条")

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果
    mock_embedding_chunks = [{'chunk_id': 466582515052979214, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979218, 'content': '产品概述：万用表RS-12是一款高精度数字万用表。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '产品概述'}, {'chunk_id': 466582515052979219, 'content': '技术参数：直流电压量程0-600V，交流电压量程0-600V。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '技术参数'}, {'chunk_id': 466582515052979220, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979221, 'content': '安全警告：测量电压时请勿超过额定输入值，防止触电。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '安全警告'}]


    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [{'chunk_id': 466582515052979214, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979220, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979226, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979232, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979239, 'content': '使用方法：旋转拨盘选择测量档位，将表笔接入被测电路。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '使用方法'}, {'chunk_id': 466582515052979221, 'content': '安全警告：测量电压时请勿超过额定输入值，防止触电。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '安全警告'}, {'chunk_id': 466582515052979218, 'content': '产品概述：万用表RS-12是一款高精度数字万用表。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '产品概述'}, {'chunk_id': 466582515052979219, 'content': '技术参数：直流电压量程0-600V，交流电压量程0-600V。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '技术参数'}, {'chunk_id': 466582515052979224, 'content': '产品概述：万用表RS-12是一款高精度数字万用表。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '产品概述'}, {'chunk_id': 466582515052979225, 'content': '技术参数：直流电压量程0-600V，交流电压量程0-600V。', 'file_title': 'test.pdf', 'item_name': '万用表RS-12', 'parent_title': 'test.pdf', 'title': '技术参数'}]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)

        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]

        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")