from langgraph.graph import StateGraph, END
from state import State
from nodes import input_node, planner_node, tool_node, fetch_node, filter_node, update_node


def build_graph():
    """Main pipeline: input → planner → tool → fetch → filter → END"""
    g = StateGraph(State)

    g.add_node("input",   input_node)
    g.add_node("planner", planner_node)
    g.add_node("tool",    tool_node)
    g.add_node("fetch",   fetch_node)
    g.add_node("filter",  filter_node)

    g.set_entry_point("input")
    g.add_edge("input",   "planner")
    g.add_edge("planner", "tool")
    g.add_edge("tool",    "fetch")
    g.add_edge("fetch",   "filter")
    g.add_edge("filter",  END)

    return g.compile()


def build_update_graph():
    """Follow-up pipeline: update → filter → END (no new web search)"""
    g = StateGraph(State)

    g.add_node("update", update_node)
    g.add_node("filter", filter_node)

    g.set_entry_point("update")
    g.add_edge("update", "filter")
    g.add_edge("filter", END)

    return g.compile()