import uuid
from typing import Optional, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import build_graph, build_update_graph
from state import State

app = FastAPI(title="Article Search Chatbot", version="2.0")

# build graphs once at startup
_search_graph = build_graph()
_update_graph = build_update_graph()

# simple in-memory session store
sessions: dict[str, State] = {}


# ── request / response models ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    search_topic:           str
    target_audience:        str   = "general"
    target_relevance_score: float = Field(0.6, ge=0.0, le=1.0)
    web_url:                Optional[str] = None

class UpdateRequest(BaseModel):
    session_id:   str
    user_message: str

class SearchResponse(BaseModel):
    session_id:    str
    final_output:  List[Any]
    article_count: int
    iteration:     int

class UpdateResponse(BaseModel):
    session_id:    str
    final_output:  List[Any]
    article_count: int
    iteration:     int


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Run a fresh article search and return results + a session_id for follow-ups."""
    initial: State = {
        "search_topic":           req.search_topic,
        "target_audience":        req.target_audience,
        "target_relevance_score": req.target_relevance_score,
        "web_url":                req.web_url or "",
        "search_results":         [],
        "fetched_articles":       [],
        "final_output":           [],
        "messages":               [],
        "user_message":           None,
        "session_id":             None,
        "iteration":              0,
    }

    try:
        result = _search_graph.invoke(initial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_id = str(uuid.uuid4())
    result["session_id"] = session_id
    sessions[session_id] = result

    output = result.get("final_output") or []
    return SearchResponse(
        session_id=session_id,
        final_output=output,
        article_count=len(output),
        iteration=result.get("iteration", 1),
    )


@app.post("/update", response_model=UpdateResponse)
def update(req: UpdateRequest):
    """
    Refine results from a previous search using a follow-up message.
    Uses the same fetched articles — no new web search needed.
    """
    state = sessions.get(req.session_id)
    if not state:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /search first."
        )

    updated = dict(state)
    updated["user_message"] = req.user_message

    try:
        result = _update_graph.invoke(updated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[req.session_id] = result

    output = result.get("final_output") or []
    return UpdateResponse(
        session_id=req.session_id,
        final_output=output,
        article_count=len(output),
        iteration=result.get("iteration", 0),
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Check status of a session."""
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id":     session_id,
        "search_topic":   state.get("search_topic"),
        "iteration":      state.get("iteration"),
        "article_count":  len(state.get("final_output") or []),
        "fetched_count":  len(state.get("fetched_articles") or []),
        "chat_turns":     len(state.get("messages") or []) // 2,
    }


@app.get("/health")
def health():
    return {"status": "ok"}