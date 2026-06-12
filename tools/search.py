from langchain_tavily import TavilySearch
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from config import get_settings
from pydantic import BaseModel, Field
import os


class TavilySearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=2, description="Maximum results to return")
    search_depth: str = Field(default="basic", description="Search depth: basic or advanced")


settings = get_settings()
os.environ["TAVILY_API_KEY"] = settings.tavily_api_key


@tool(args_schema=TavilySearchInput)
def tavily_search_tool(query: str, max_results: int = 2, search_depth: str = "basic") -> list[dict]:
    """
    Search the web using Tavily API for current information.
    Returns list of results with title, url, content, and score.
    """
    search = TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
    )
    return search.invoke({"query": query})
