from langgraph.graph import StateGraph
from state import State
from nodes import *

def build_graph():
    builder = StateGraph(State)
    builder.add_node("prepare", prepare_input_node)
    builder.add_node("search", search_node)
    builder.add_node("fetch", fetch_articles_node)
    builder.add_node("filter", filter_articles_node)

    builder.set_entry_point("prepare")

    builder.add_edge("prepare", "search")
    builder.add_edge("search", "fetch")
    builder.add_edge("fetch", "filter")

    return builder.compile()