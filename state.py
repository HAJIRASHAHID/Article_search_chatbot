from typing import TypedDict, List, Optional


class Article(TypedDict):
    title: str
    url: str
    relevance_score: float
    full_content: str
    suggested_topic: str


class State(TypedDict):
    # user inputs
    search_topic: str
    target_audience: str
    target_relevance_score: float
    web_url: Optional[str]

    # pipeline data
    search_results: List[dict]
    fetched_articles: List[dict]
    final_output: List[Article]

    # conversation memory for multi-turn
    messages: List[dict]
    user_message: Optional[str]
    session_id: Optional[str]
    iteration: int