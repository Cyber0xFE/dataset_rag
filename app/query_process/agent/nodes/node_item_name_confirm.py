import json
import sys
from uuid import uuid4

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.lm import lm_utils
from app.conf.lm_config import lm_config
from app.clients.milvus_utils import get_milvus_client, hybrid_search, create_hybrid_search_requests
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings


def _extract_item_names(query: str, history_text: str):
    """调用 LLM 提取商品名称并改写问题。"""
    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
    llm = lm_utils.get_llm_client(lm_config.llm_model, json_mode=True)
    resp = llm.invoke([
        {"role": "system", "content": "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"},
        {"role": "user", "content": prompt},
    ])
    content = resp.content.strip()

    try:
        clean = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        item_names = result.get("item_names", [])
        rewritten_query = result.get("rewritten_query", query)
        logger.info(f"提取商品名称: {item_names}")
        return item_names, rewritten_query
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"LLM 返回解析失败: {e}\n原始内容: {content}")
        return [], query


def _match_item_names(item_names: list[str]) -> list[dict]:
    """将 item_names 向量化后与 Milvus 匹配，返回匹配结果列表。"""
    if not item_names:
        return []

    client = get_milvus_client()
    if not client:
        logger.warning("Milvus 未连接，跳过向量匹配")
        return [{"extracted_name": n, "matches": []} for n in item_names]

    if not client.has_collection(milvus_config.item_name_collection):
        logger.warning(f"Milvus 集合 {milvus_config.item_name_collection} 不存在")
        return [{"extracted_name": n, "matches": []} for n in item_names]

    client.load_collection(milvus_config.item_name_collection)

    try:
        embeddings = generate_embeddings(item_names)
        dense_vecs = embeddings["dense"]
        sparse_vecs = embeddings["sparse"]

        result = []
        for i, name in enumerate(item_names):
            reqs = create_hybrid_search_requests(dense_vecs[i], sparse_vecs[i], limit=5)
            res = hybrid_search(client, milvus_config.item_name_collection, reqs, ranker_weights=(0.4, 0.6),
                                norm_score=True, limit=5, output_fields=["item_name"])
            matches = []
            if res and res[0]:
                for hit in res[0]:
                    matched_name = hit["entity"].get("item_name", "")
                    score = hit.get("distance", None)
                    matches.append({"item_name": matched_name, "score": score})
            result.append({"extracted_name": name, "matches": matches})
        return result
    except Exception as e:
        logger.error(f"商品匹配异常: {e}")
        return [{"extracted_name": n, "matches": []} for n in item_names]


def _filter_matches(matched_items: list[dict]) -> tuple[list[str], list[dict]]:
    """筛选匹配结果，返回 (confirmed_item_names, options)。"""
    confirmed_item_names = []
    options = []
    for item in matched_items:
        high = [m for m in item["matches"] if m.get("score") is not None and m["score"] >= 0.85]
        if high:
            exact = [m for m in high if m["item_name"] == item["extracted_name"]]
            if exact:
                confirmed_item_names.append(exact[0]["item_name"])
            else:
                confirmed_item_names.append(max(high, key=lambda m: m["score"])["item_name"])
        else:
            mid = [m for m in item["matches"] if m.get("score") is not None and 0.6 <= m["score"] < 0.85]
            mid.sort(key=lambda m: m["score"], reverse=True)
            options.extend(mid[:2])
    return confirmed_item_names, options


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names'], state['rewritten_query']
    """
    logger.info("---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    # 从mongo获取历史对话
    history = get_recent_messages(state["session_id"], limit=10)
    state["history"] = history
    # 保存当前用户消息
    save_chat_message(state["session_id"], "user", state["original_query"])

    # 历史对话转文本
    history_lines = []
    for msg in history:
        role = "用户" if msg.get("role") == "user" else "助手"
        text = msg.get("text", "")
        history_lines.append(f"{role}: {text}")
    history_text = "\n".join(history_lines) if history_lines else "无历史对话"

    # 调用 LLM 提取商品名称和改写问题
    item_names, rewritten_query = _extract_item_names(state["original_query"], history_text)

    # 向量化后与 Milvus 匹配确认
    matched_items = _match_item_names(item_names)

    # 筛选匹配结果
    confirmed_item_names, options = _filter_matches(matched_items)
    logger.info(f"已确认商品: {confirmed_item_names}, 候选: {[m['item_name'] for m in options]}")

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    logger.info("---node_item_name_confirm---处理结束")

    return {
        "rewritten_query": rewritten_query,
        "item_names": confirmed_item_names,
        "matched_items": matched_items,
        "options": options,
    }


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": str(uuid4()),
        "original_query": "RS-12万用表怎么用？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
