from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from state import State

app = FastAPI()

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search_articles(state: State):
    state = user_input_node(state)
    state = web_search_node(state)
    state = fetch_full_articles(state)
    state = filter_and_score_node(state)
    return state["final_articles"]

@app.post("/search")
def search_articles(state: State):
    state = user_input_node(state)
    state = web_search_node(state)
    state = fetch_full_articles_node(state)
    state = filter_and_score_node(state)
    return state["final_articles"]