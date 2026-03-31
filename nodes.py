import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from state import State
from tools import web_search
from utils import fetch_full_content

# ── two models: fast for planning, strong for filtering ──────────────────────
llm_fast   = ChatGroq(model="llama-3.1-8b-instant",     temperature=0, api_key=os.getenv("GROQ_API_KEY"))
llm_strong = ChatGroq(model="llama-3.3-70b-versatile",  temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# bind web_search tool to fast LLM so planner can call it
@tool
def search_tool(query: str) -> str:
    """Search the web for articles about the given query."""
    results = web_search(query)
    return json.dumps(results)

llm_with_tools = llm_fast.bind_tools([search_tool], tool_choice="auto")


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 – input_node
# ─────────────────────────────────────────────────────────────────────────────
def input_node(state: State) -> dict:
    return {
        "search_topic":           state.get("search_topic", "").strip(),
        "target_audience":        state.get("target_audience", "general"),
        "target_relevance_score": float(state.get("target_relevance_score", 0.6)),
        "web_url":                state.get("web_url", ""),
        "search_results":         [],
        "fetched_articles":       [],
        "final_output":           [],
        "messages":               state.get("messages", []),
        "user_message":           state.get("user_message"),
        "session_id":             state.get("session_id"),
        "iteration":              state.get("iteration", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 – planner_node  (LLM call 1 — decides what to search)
# ─────────────────────────────────────────────────────────────────────────────
def planner_node(state: State) -> dict:
    topic    = state["search_topic"]
    audience = state["target_audience"]

    messages = [
        SystemMessage(content=(
            f"You are a research assistant. "
            f"Your ONLY job is to call the search_tool to find articles about '{topic}' "
            f"for a {audience} audience. Always call the tool."
        )),
        HumanMessage(content=f"Search for articles about: {topic}"),
    ]

    try:
        response = llm_with_tools.invoke(messages)
        return {"planner_response": response}
    except Exception as e:
        print(f"[planner_node] error: {e}")
        return {"planner_response": None}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 – tool_node  (executes the tool call from planner)
# ─────────────────────────────────────────────────────────────────────────────
def tool_node(state: State) -> dict:
    response = state.get("planner_response")
    results  = []

    # try to use tool call from LLM
    if response and hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            query = tc.get("args", {}).get("query", state["search_topic"])
            print(f"[tool_node] searching: {query}")
            raw = search_tool.invoke(query)
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(data, list):
                    results.extend(data)
            except Exception:
                pass

    # fallback: LLM skipped the tool, search directly
    if not results:
        query = f"{state['search_topic']} {state['target_audience']}"
        print(f"[tool_node] fallback search: {query}")
        results = web_search(query)

    print(f"[tool_node] got {len(results)} results")
    return {"search_results": results}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 – fetch_node  (download full article text)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_node(state: State) -> dict:
    fetched = []

    for item in state.get("search_results", []):
        url     = (item.get("url") or "").strip()
        title   = item.get("title", "Untitled")
        snippet = item.get("content") or item.get("raw_content") or ""

        if not url:
            continue

        # try newspaper3k, fall back to Tavily snippet
        full_text = fetch_full_content(url)
        if len(full_text) < 200:
            full_text = snippet

        if len(full_text) < 100:
            print(f"[fetch_node] skipping (no content): {url}")
            continue

        fetched.append({"title": title, "url": url, "full_content": full_text})

    print(f"[fetch_node] fetched {len(fetched)} articles")
    return {"fetched_articles": fetched}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 – filter_node  (LLM call 2 — score and filter, return strict JSON)
# ─────────────────────────────────────────────────────────────────────────────
def filter_node(state: State) -> dict:
    fetched   = state.get("fetched_articles", [])
    threshold = float(state.get("target_relevance_score", 0.6))
    topic     = state["search_topic"]
    audience  = state["target_audience"]

    if not fetched:
        print("[filter_node] no articles to filter")
        return {"final_output": [], "iteration": state.get("iteration", 0) + 1}

    # cap content so we don't blow the context window
    payload = [
        {"title": a["title"], "url": a["url"], "full_content": a["full_content"][:3000]}
        for a in fetched
    ]

    # build message history for context-aware filtering
    history_msgs = _build_history(state.get("messages", []))

    system = SystemMessage(content="""You are an article filtering assistant.

STRICT RULES — follow exactly:
1. Output ONLY a valid JSON array. No explanation, no markdown, no code fences.
2. Each item must have: title, url, relevance_score (float 0-1), full_content, suggested_topic
3. Only include articles that meet the minimum relevance_score
4. If none qualify, return empty array: []""")

    user_content = (
        f"Topic: {topic}\nAudience: {audience}\nMin score: {threshold}\n\n"
        f"Articles:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    human = HumanMessage(content=user_content)

    try:
        response = llm_strong.invoke([system] + history_msgs + [human])
        raw_text = response.content.strip()
    except Exception as e:
        print(f"[filter_node] LLM error: {e}")
        return {"final_output": [], "iteration": state.get("iteration", 0) + 1}

    final_output = _parse_json(raw_text)

    # save this turn to conversation history
    updated_messages = list(state.get("messages", [])) + [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": raw_text},
    ]

    print(f"[filter_node] kept {len(final_output)} articles")
    return {
        "final_output": final_output,
        "messages":     updated_messages,
        "iteration":    state.get("iteration", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 – update_node  (re-entry for follow-up messages)
# Re-filters existing fetched_articles without a new web search
# ─────────────────────────────────────────────────────────────────────────────
def update_node(state: State) -> dict:
    user_msg  = (state.get("user_message") or "").strip()
    fetched   = state.get("fetched_articles", [])
    threshold = float(state.get("target_relevance_score", 0.6))

    if not user_msg:
        return {}

    # let user change threshold inline e.g. "raise score to 0.8"
    nums = re.findall(r"\b0\.\d+\b", user_msg)
    if nums:
        threshold = float(nums[0])
        print(f"[update_node] new threshold: {threshold}")

    payload = [
        {"title": a["title"], "url": a["url"], "full_content": a["full_content"][:3000]}
        for a in fetched
    ]

    history_msgs = _build_history(state.get("messages", []))

    system = SystemMessage(content="""You are an article filtering assistant.

STRICT RULES:
1. Output ONLY a valid JSON array. No explanation, no markdown, no code fences.
2. Each item must have: title, url, relevance_score, full_content, suggested_topic
3. Apply the user's instructions when re-filtering
4. If none qualify, return: []""")

    user_content = (
        f"User request: {user_msg}\n\n"
        f"Topic: {state['search_topic']}\nAudience: {state['target_audience']}\n"
        f"Min score: {threshold}\n\n"
        f"Re-filter these articles:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    human = HumanMessage(content=user_content)

    try:
        response = llm_strong.invoke([system] + history_msgs + [human])
        raw_text = response.content.strip()
    except Exception as e:
        print(f"[update_node] LLM error: {e}")
        return {}

    final_output = _parse_json(raw_text)

    updated_messages = list(state.get("messages", [])) + [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": raw_text},
    ]

    print(f"[update_node] updated: {len(final_output)} articles")
    return {
        "final_output":           final_output,
        "messages":               updated_messages,
        "target_relevance_score": threshold,
        "iteration":              state.get("iteration", 0) + 1,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> list:
    """Extract JSON array from LLM output, handles markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        for key in ("articles", "results", "output"):
            if isinstance(data, dict) and key in data:
                return data[key]
    except json.JSONDecodeError:
        pass

    # last resort: find first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _build_history(messages: list) -> list:
    """Convert stored message dicts to LangChain message objects (last 6 only)."""
    out = []
    for m in messages[-6:]:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out