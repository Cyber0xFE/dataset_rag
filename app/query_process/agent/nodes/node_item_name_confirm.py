import json
import sys

from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.lm import lm_utils
from app.conf.lm_config import lm_config


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


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names'], state['rewritten_query']
    """
    logger.info("---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

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

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    logger.info("---node_item_name_confirm---处理结束")

    return {"item_names": item_names, "rewritten_query": rewritten_query}


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用22？",
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