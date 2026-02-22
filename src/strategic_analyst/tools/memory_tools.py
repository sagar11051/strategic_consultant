"""
Memory tools for reading and updating user memory namespaces.

These are *stub* tools: the LLM sees their docstrings and Pydantic schemas for
tool-selection purposes, but the actual store access happens inside make_tool_node()
in base.py, which has both `store` and `user_id` in closure scope.

Calling either tool directly (outside of make_tool_node) raises NotImplementedError.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ── memory_search_tool ────────────────────────────────────────────────────────

class MemorySearchInput(BaseModel):
    namespace_type: Literal[
        "user_profile",
        "company_profile",
        "user_preferences",
        "episodic_memory",
    ] = Field(description="Which memory namespace to search")
    query: str = Field(
        description="What you are looking for in this memory namespace"
    )


@tool(args_schema=MemorySearchInput)
def memory_search_tool(namespace_type: str, query: str) -> str:
    """
    Search the user's personal memory for relevant context.

    Namespaces:
    - user_profile:      who the user is, their role, communication style
    - company_profile:   company context, industry, competitors, vocabulary
    - user_preferences:  preferred report format, frameworks, verbosity, citation style
    - episodic_memory:   past research sessions, important dates, temporal notes

    Always check memory before making assumptions about the user or their company.
    """
    raise NotImplementedError(
        "memory_search_tool must be invoked via make_tool_node(), "
        "which injects store + user_id at runtime."
    )


# ── write_memory_tool ─────────────────────────────────────────────────────────

class WriteMemoryInput(BaseModel):
    namespace_type: Literal[
        "user_profile",
        "company_profile",
        "user_preferences",
        "episodic_memory",
    ] = Field(description="Which memory namespace to update")
    update_reason: str = Field(
        description=(
            "Explain what new information was discovered and why this memory "
            "should be updated"
        )
    )
    context: str = Field(
        description=(
            "The new information or feedback that should be incorporated into memory"
        )
    )


@tool(args_schema=WriteMemoryInput)
def write_memory_tool(namespace_type: str, update_reason: str, context: str) -> str:
    """
    Update a user memory namespace with new information discovered during research
    or revealed through user feedback.

    Use this when:
    - User reveals personal information               (-> user_profile)
    - Research uncovers new company context           (-> company_profile)
    - User gives style or format feedback             (-> user_preferences)
    - A session concludes with notable temporal notes (-> episodic_memory)

    IMPORTANT: This makes a targeted LLM-driven update — it never overwrites existing
    memory, only integrates the new information into what is already stored.
    """
    raise NotImplementedError(
        "write_memory_tool must be invoked via make_tool_node(), "
        "which injects store + user_id at runtime."
    )
