from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import (
    node_item_name_recognition,
)
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState


def route_entry(state: ImportGraphState) -> str:
    if state.is_md_read_enabled:
        return "to_node_md_img"
    elif state.is_pdf_read_enabled:
        return "to_node_pdf_to_md"
    return "to_end"


graph = StateGraph(ImportGraphState)

graph.add_node("node_entry", node_entry)
graph.add_node("node_pdf_to_md", node_pdf_to_md)
graph.add_node("node_md_img", node_md_img)
graph.add_node("node_document_split", node_document_split)
graph.add_node("node_item_name_recognition", node_item_name_recognition)
graph.add_node("node_bge_embedding", node_bge_embedding)
graph.add_node("node_import_milvus", node_import_milvus)

graph.add_edge(START, "node_entry")
graph.add_conditional_edges("node_entry", route_entry, {
    "to_node_md_img": "node_md_img",
    "to_node_pdf_to_md": "node_pdf_to_md",
    "to_end": END,
})

graph.add_edge("node_pdf_to_md", "node_md_img")
graph.add_edge("node_md_img", "node_document_split")
graph.add_edge("node_document_split", "node_item_name_recognition")
graph.add_edge("node_item_name_recognition", "node_bge_embedding")
graph.add_edge("node_bge_embedding", "node_import_milvus")
graph.add_edge("node_import_milvus", END)

app = graph.compile()
app.get_graph().print_ascii()
