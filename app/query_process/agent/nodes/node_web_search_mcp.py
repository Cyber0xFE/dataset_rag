import json
import sys

from app.core.logger import logger
from app.clients.mcp_client import get_mcp_client
from app.utils.task_utils import add_done_task, add_running_task


def mcp_call(query: str, count: int = 10) -> list[dict]:
    """调用百炼 MCP bailian_web_search 服务，返回搜索结果列表。"""
    client = get_mcp_client()
    result = client.call_tool("bailian_web_search", {"query": query, "count": count})
    # MCP 返回格式：result.content[{type, text}]，text 为 JSON 字符串
    content = result.get("content", []) if isinstance(result, dict) else []
    items = []
    for c in content:
        if c.get("type") == "text":
            try:
                data = json.loads(c["text"])
                pages = data.get("pages", []) if isinstance(data, dict) else []
                items.extend(pages)
            except (json.JSONDecodeError, KeyError):
                items.append({"text": c["text"]})
    logger.info(f"MCP 网络搜索到 {len(items)} 条结果")
    return items


def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息。
    输入：state['rewritten_query']
    输出：更新 state['web_search_docs']
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    query = state.get("rewritten_query") or state.get("original_query", "")
    if query:
        results = mcp_call(query)
        state["web_search_docs"] = results
    else:
        logger.warning("查询为空，跳过网络搜索")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state


if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")