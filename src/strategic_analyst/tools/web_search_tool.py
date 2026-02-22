"""
Web search tool using Tavily.

Wraps TavilyClient in an asyncio executor so it is safe to await inside
async LangGraph node functions without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query")
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return (1–10)",
    )
    search_depth: str = Field(
        default="advanced",
        description="Search depth: 'basic' for quick results, 'advanced' for deeper analysis",
    )


@tool(args_schema=WebSearchInput)
async def web_search_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> str:
    """
    Search the web for current market intelligence, competitor information,
    industry trends, news, and publicly available strategic data.

    Use this when:
    - The company knowledge base lacks sufficient or up-to-date information
    - You need current events, recent news, or real-time market data
    - You want to validate internal findings against external sources

    Returns a quick-answer summary (if available) followed by numbered results
    with title, URL, and content excerpt. Always cite URLs in your response.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        ),
    )

    parts: list[str] = []
    if response.get("answer"):
        parts.append(f"**Quick Answer:** {response['answer']}\n")
    for i, result in enumerate(response.get("results", []), 1):
        parts.append(
            f"[{i}] **{result.get('title', 'Untitled')}**\n"
            f"URL: {result.get('url', 'No URL')}\n"
            f"{result.get('content', 'No content')}\n"
        )

    return "\n---\n".join(parts) if parts else "No web search results found."
