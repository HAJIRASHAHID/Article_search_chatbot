import os
import json
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper


def web_search(query: str) -> list:
    try:
        tavily = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
        results = tavily.results(query, max_results=8)
        return results if isinstance(results, list) else []
    except Exception as e:
        print(f"[web_search] error: {e}")
        return []