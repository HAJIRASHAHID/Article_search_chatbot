from langgraph.graph import StateGraph
from state import State
from nodes import (
    user_input_node,
    web_search_node,
    fetch_full_articles_node,
    filter_and_score_node
)

def build_graph():
    """
    Builds the LangGraph computation graph for the article search pipeline.
    """
    builder = StateGraph(State)
    
    # Add nodes with CORRECT function names from nodes.py
    builder.add_node("prepare", user_input_node)
    builder.add_node("search", web_search_node)
    builder.add_node("fetch", fetch_full_articles_node)
    builder.add_node("filter", filter_and_score_node)

    # Set entry point
    builder.set_entry_point("prepare")

    # Define edges (transitions between nodes)
    builder.add_edge("prepare", "search")
    builder.add_edge("search", "fetch")
    builder.add_edge("fetch", "filter")

    # Compile and return the graph
    return builder.compile()

# Create the graph instance
app_graph = build_graph()