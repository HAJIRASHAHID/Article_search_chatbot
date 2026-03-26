from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

tavily = TavilySearchAPIWrapper()

def web_search(query: str):
    """Search the web for articles related to a query."""
    try:
        return tavily.results(query, max_results=5)
    except Exception as e:
        print("WEB SEARCH ERROR:", e)
        return []