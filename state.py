from typing import List, Optional, Dict
from typing_extensions import TypedDict  # must use typing_extensions for Python < 3.12

class Article(TypedDict):
    title: str
    url: str
    relevance_score: float
    full_content: str
    suggested_topic: str

class State(TypedDict, total=False):
    search_topic: str
    target_audience: Optional[str]
    target_relevance_score: Optional[float]
    web_url: Optional[str]
    search_query: Optional[str]
    search_results: Optional[List[Dict]]
    fetched_articles: Optional[List[Article]]
    final_articles: Optional[List[Article]]