from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import os
import json

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================
# Define web_search as a langchain tool
# ============================================
@tool
def web_search(query: str) -> str:
    """
    Search the web for articles related to a query.
    
    Args:
        query: The search query string
        
    Returns:
        JSON string containing search results
    """
    try:
        tavily = TavilySearchAPIWrapper()
        results = tavily.results(query, max_results=5)
        return json.dumps(results)
    except Exception as e:
        print(f"WEB SEARCH ERROR: {e}")
        return json.dumps({"error": str(e), "results": []})


# ============================================
# Bind tools to LLM with tool_choice="auto"
# ============================================
tools = [web_search]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")