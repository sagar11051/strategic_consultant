"""
Tool registry and node factory for the strategic analyst agent.

Exports:
  ALL_TOOLS_REGISTRY      — dict of all tool objects keyed by name
  *_TOOLS                 — per-agent tool-name lists
  get_tools()             — resolve names -> tool objects
  get_tools_by_name()     — list[BaseTool] -> dict keyed by name
  make_tool_node()        — async tool execution node with store injection
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from strategic_analyst.tools.rag_tool import semantic_search, hybrid_search
from strategic_analyst.tools.web_search_tool import web_search_tool
from strategic_analyst.tools.memory_tools import memory_search_tool, write_memory_tool
from strategic_analyst.tools.question_tool import Question, Done


# ── Per-agent tool-name lists ──────────────────────────────────────────────────
# Use hybrid_search as the default search; semantic_search for conceptual-only queries.

MAIN_AGENT_TOOLS: List[str] = [
    "semantic_search",
    "hybrid_search",
    "memory_search_tool",
]

PLANNER_TOOLS: List[str] = [
    "semantic_search",
    "hybrid_search",
    "memory_search_tool",
]

RESEARCH_SUPERVISOR_TOOLS: List[str] = [
    "semantic_search",
    "hybrid_search",
    "web_search_tool",
    "memory_search_tool",
    "write_memory_tool",
    "Question",
]

TASK_AGENT_TOOLS: List[str] = [
    "semantic_search",
    "hybrid_search",
    "web_search_tool",
    "Done",
]

REPORT_SUPERVISOR_TOOLS: List[str] = [
    "memory_search_tool",
    "write_memory_tool",
    "Question",
]

REPORT_WRITER_TOOLS: List[str] = [
    "semantic_search",
    "hybrid_search",
    "memory_search_tool",
    "Done",
]


# ── Registry ───────────────────────────────────────────────────────────────────

ALL_TOOLS_REGISTRY: Dict[str, BaseTool] = {
    "semantic_search":    semantic_search,
    "hybrid_search":      hybrid_search,
    "web_search_tool":    web_search_tool,
    "memory_search_tool": memory_search_tool,
    "write_memory_tool":  write_memory_tool,
    "Question":           Question,
    "Done":               Done,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:
    """
    Return tool objects by name.
    If tool_names is None, returns every tool in the registry.
    Unknown names are silently skipped.
    """
    if tool_names is None:
        return list(ALL_TOOLS_REGISTRY.values())
    return [ALL_TOOLS_REGISTRY[name] for name in tool_names if name in ALL_TOOLS_REGISTRY]


def get_tools_by_name(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    """Convert a list of tool objects to a name-keyed dict for fast lookup."""
    return {t.name: t for t in tools}


# ── Tool node factory ─────────────────────────────────────────────────────────

def make_tool_node(tools: List[BaseTool], store: BaseStore, user_id: str):
    """
    Create an async LangGraph tool-execution node with store injection.

    Dispatch rules:
      memory_search_tool  → calls search_memory(store, user_id, ...)
      write_memory_tool   → calls update_memory_with_llm_async(store, ...)
      Question / Done     → signal acknowledgement only (no side effects)
      everything else     → tool.ainvoke(args)

    Returns a coroutine function ``tool_node(state) -> {"messages": [...]}``.
    """
    tools_by_name = get_tools_by_name(tools)

    # Import here to avoid circular imports at module load time
    from strategic_analyst.memory import (
        search_memory,
        update_memory_with_llm_async,
        user_profile_ns,
        company_profile_ns,
        user_preferences_ns,
        episodic_memory_ns,
        NAMESPACE_KEYS,
    )

    _namespace_map: Dict[str, tuple] = {
        "user_profile":    (user_profile_ns(user_id),     NAMESPACE_KEYS["user_profile"]),
        "company_profile": (company_profile_ns(user_id),  NAMESPACE_KEYS["company_profile"]),
        "user_preferences":(user_preferences_ns(user_id), NAMESPACE_KEYS["user_preferences"]),
        "episodic_memory": (episodic_memory_ns(user_id),  NAMESPACE_KEYS["episodic_memory"]),
    }

    async def tool_node(state: dict) -> dict:
        results: list[dict] = []

        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            args      = tool_call["args"]
            call_id   = tool_call["id"]

            # ── Memory read ──────────────────────────────────────────────────
            if tool_name == "memory_search_tool":
                ns_type = args.get("namespace_type", "user_profile")
                content = search_memory(store, user_id, ns_type, args.get("query", ""))
                results.append({"role": "tool", "content": content, "tool_call_id": call_id})

            # ── Memory write ─────────────────────────────────────────────────
            elif tool_name == "write_memory_tool":
                ns_type = args.get("namespace_type", "user_preferences")
                namespace, key = _namespace_map[ns_type]
                await update_memory_with_llm_async(
                    store=store,
                    namespace=namespace,
                    key=key,
                    context_messages=[{"role": "user", "content": args.get("context", "")}],
                    update_reason=args.get("update_reason", "Agent-initiated memory update"),
                )
                results.append({
                    "role": "tool",
                    "content": f"Memory namespace '{ns_type}' updated successfully.",
                    "tool_call_id": call_id,
                })

            # ── Signal tools (no side effects) ───────────────────────────────
            elif tool_name in ("Question", "Done"):
                results.append({
                    "role": "tool",
                    "content": f"Signal '{tool_name}' received.",
                    "tool_call_id": call_id,
                })

            # ── Regular tools (rag, web search) ──────────────────────────────
            else:
                tool_obj = tools_by_name[tool_name]
                observation = await tool_obj.ainvoke(args)
                results.append({
                    "role": "tool",
                    "content": str(observation),
                    "tool_call_id": call_id,
                })

        return {"messages": results}

    return tool_node
