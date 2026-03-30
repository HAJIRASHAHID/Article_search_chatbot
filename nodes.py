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
def web_search_node(state: State) -> State:  # Uses TavilySearchAPIWrapper to perform a web search using the search_query. Stores results in search_results key of state.
    """
    Calls Tavily web search to get top search results.
    """
    try:
        tavily = TavilySearchAPIWrapper()
        results = tavily.results(state["search_query"], max_results=5)
        state["search_results"] = results
    except Exception as e:
        print(f"Web search failed: {e}")
        state["search_results"] = []
    return state

# -----------------------------
# Node 3: Fetch Full Articles Node
# Why it exists: Search results often only contain snippets. This node fetches the full article content for better relevance scoring.
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
                "relevance_score": 0.0,
                "full_content": news.text,
                "suggested_topic": state["search_topic"]
            })
        except Exception as e:
            print(f"Failed to fetch article {result}: {e}")
    state["fetched_articles"] = articles
    return state

# -----------------------------
# Node 4: Filter & Score Node   
# #why it exists:Not all articles from the web are useful. This node filters and ranks articles to return only the most relevant ones.
# -----------------------------
def filter_and_score_node(state: State) -> State:
    """
    Filters articles by relevance score and assigns final scores.
    """
    final_articles: List[Article] = []
    for article in state["fetched_articles"]:
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
# Why it exists: Provides a single function to run the entire pipeline in order, making it easier to call from the API or other interfaces. 
# -----------------------------
def run_pipeline(state: State) -> List[Article]:
    """
    Executes all nodes in order and returns final articles.
    """
    state = user_input_node(state) #prepare input
    state = web_search_node(state) #fetch urls
    state = fetch_full_articles_node(state) #fetch full content
    state = filter_and_score_node(state) #filter and score
    return state["final_articles"] #return final articles