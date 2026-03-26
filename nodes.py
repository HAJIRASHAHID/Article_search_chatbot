# nodes.py
from typing import List
from newspaper import Article as NewsArticle
from state import State, Article
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# -----------------------------
# Node 1: User Input Node
# -----------------------------
def user_input_node(state: State) -> State:
    """
    Processes user input and prepares search query.
    """
    state["search_topic"] = state.get("search_topic", "").strip()
    state["target_audience"] = state.get("target_audience", "").strip()
    state["target_relevance_score"] = state.get("target_relevance_score", 0.7)
    state["web_url"] = state.get("web_url", None)

    # Generate search query automatically if not provided
    state["search_query"] = state.get("search_query") or f"articles on {state['search_topic']} for {state['target_audience']}"
    return state

# -----------------------------
# Node 2: Web Search Node
# -----------------------------
def web_search_node(state: State) -> State:
    """
    Calls Tavily web search to get top search results.
    """
    try:
        tavily = TavilySearchAPIWrapper()
        results = tavily.run(state["search_query"])
        state["search_results"] = results
    except Exception as e:
        print(f"Web search failed: {e}")
        state["search_results"] = []
    return state

# -----------------------------
# Node 3: Fetch Full Articles Node
# -----------------------------
def fetch_full_articles_node(state: State) -> State:
    """
    Extracts full article content using newspaper3k.
    """
    articles: List[Article] = []
    for result in state["search_results"]:
        try:
            url = result.get("link") or result.get("url")
            title = result.get("title", "No Title")
            news = NewsArticle(url)
            news.download()
            news.parse()
            articles.append({
                "title": title,
                "url": url,
                "relevance_score": 0.0,  # will be updated in filter node
                "full_content": news.text,
                "suggested_topic": state["search_topic"]
            })
        except Exception as e:
            print(f"Failed to fetch article {result}: {e}")
    state["fetched_articles"] = articles
    return state

# -----------------------------
# Node 4: Filter & Score Node
# -----------------------------
def filter_and_score_node(state: State) -> State:
    """
    Filters articles by relevance score and assigns final scores.
    Replace with LLM logic for real filtering.
    """
    final_articles: List[Article] = []
    for article in state["fetched_articles"]:
        # Dummy scoring: you can replace this with LLM call
        score = 0.9
        if score >= state["target_relevance_score"]:
            final_articles.append({
                "title": article["title"],
                "url": article["url"],
                "relevance_score": score,
                "full_content": article["full_content"],
                "suggested_topic": article["suggested_topic"]
            })
    state["final_articles"] = final_articles
    return state

# -----------------------------
# Node 5: Full Pipeline
# -----------------------------
def run_pipeline(state: State) -> List[Article]:
    """
    Executes all nodes in order and returns final articles.
    """
    state = user_input_node(state)
    state = web_search_node(state)
    state = fetch_full_articles_node(state)
    state = filter_and_score_node(state)
    return state["final_articles"]