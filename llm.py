from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from tools import web_search
import os

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Search the web for articles related to a query."
)

llm_with_tools = llm.bind_tools([search_tool])