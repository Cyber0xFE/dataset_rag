from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.query_process.agent.QueryGraphState import QueryGraphState
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_answer_output import node_answer_output


def _dummy_node(state: QueryGraphState) -> dict:
    """虚拟代理节点，仅用于多路召回扇出。"""
    return {}


graph = StateGraph(QueryGraphState)

# 添加节点
graph.add_node("node_item_name_confirm", node_item_name_confirm)
graph.add_node("node_multi_recall", _dummy_node)
graph.add_node("node_search_embedding", node_search_embedding)
graph.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
graph.add_node("node_web_search_mcp", node_web_search_mcp)
# graph.add_node("node_query_kg", node_query_kg)
graph.add_node("node_rrf", node_rrf)
graph.add_node("node_rerank", node_rerank)
graph.add_node("node_answer_output", node_answer_output)

# STAR → 1.意图识别与改写
graph.add_edge(START, "node_item_name_confirm")


def route_from_confirm(state: QueryGraphState) -> str:
    """state 中已有 answer 则直接跳到输出，否则走多路召回。"""
    return "node_answer_output" if state.get("answer") else "node_multi_recall"


# 1 → 条件边（有 answer 直接输出，否则走虚拟代理节点）
graph.add_conditional_edges("node_item_name_confirm", route_from_confirm, {
    "node_multi_recall": "node_multi_recall",
    "node_answer_output": "node_answer_output",
})

# 虚拟代理 → 三路召回（并行扇出）
graph.add_edge("node_multi_recall", "node_search_embedding")
graph.add_edge("node_multi_recall", "node_search_embedding_hyde")
graph.add_edge("node_multi_recall", "node_web_search_mcp")

# 三路召回 → 3.结果融合与粗排
graph.add_edge("node_search_embedding", "node_rrf")
graph.add_edge("node_search_embedding_hyde", "node_rrf")
graph.add_edge("node_web_search_mcp", "node_rrf")

# 3 → 4.重排序 → 5.LLM生成答案 → END
graph.add_edge("node_rrf", "node_rerank")
graph.add_edge("node_rerank", "node_answer_output")
graph.add_edge("node_answer_output", END)

kb_query_app = graph.compile()

if __name__ == "__main__":
    kb_query_app.get_graph().print_ascii()
