from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from state import State
from nodes import (
    user_input_node,
    web_search_node, 
    fetch_full_articles_node,
    filter_and_score_node,
    run_pipeline
)

app = FastAPI()

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search_articles(state: State):
    """
    Single endpoint that runs the complete pipeline:
    1. User Input Processing
    2. Web Search
    3. Fetch Full Articles
    4. Filter & Score
    """
    try:
        state = user_input_node(state)
        state = web_search_node(state)
        state = fetch_full_articles_node(state)
        state = filter_and_score_node(state)
        return {"status": "success", "articles": state.get("final_articles", [])}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/pipeline")
async def run_full_pipeline(search_topic: str, target_audience: str, target_relevance_score: float = 0.7):
    """
    Alternative endpoint that accepts query parameters
    """
    try:
        state = State()
        state["search_topic"] = search_topic
        state["target_audience"] = target_audience
        state["target_relevance_score"] = target_relevance_score
        
        final_articles = run_pipeline(state)
        return {"status": "success", "articles": final_articles}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok"}