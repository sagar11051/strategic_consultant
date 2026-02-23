"""
context_loader.py — Session initialisation node.

Fires parallel async calls to:
  1. Load all four memory namespaces for the user
  2. Execute 3 hybrid_search queries based on the user's query

Does NOT call an LLM — purely data loading.

Also contains greeting_node which runs after context is loaded.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.memory import load_all_memory
from strategic_analyst.prompts import GREETING_SYSTEM_PROMPT
from strategic_analyst.schemas import AgentState
from strategic_analyst.tools.rag_tool import hybrid_search


async def context_loader(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Session initialisation node — runs before anything else.

    Parallel actions:
      - Load all four memory namespaces (user_profile, company_profile,
        user_preferences, episodic_memory)
      - Fire 3 hybrid_search queries derived from the user's query

    Returns a full state init dict.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    # Fallback chain: configurable > AgentInput state field > hardcoded default.
    # Studio may pass user_id="" when the field is left blank — treat that as unset.
    user_id = cfg.user_id or state.get("user_id") or "default_user"
    session_id = cfg.session_id or str(uuid.uuid4())
    query = state.get("query", "")

    # Three query variants: direct, broadened, company-scoped
    per_query_limit = max(3, cfg.rag_top_k // 3)
    rag_queries = [
        query,
        f"strategic context: {query}",
        f"company data related to: {query}",
    ]

    # Memory load runs in a thread (sync store operations)
    memory_task = asyncio.create_task(
        asyncio.to_thread(load_all_memory, store, user_id)
    )

    # RAG tasks run concurrently
    rag_tasks = [
        asyncio.create_task(
            hybrid_search.ainvoke({"query_text": q, "limit": per_query_limit})
        )
        for q in rag_queries
        if q.strip()
    ]

    memory_context = await memory_task
    rag_results = await asyncio.gather(*rag_tasks, return_exceptions=True)

    # Combine RAG results; skip failed calls gracefully
    retrieved_chunks: list[dict] = []
    for result in rag_results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, str) and result.strip():
            retrieved_chunks.append({"content": result, "source": "company_db"})

    # Seed identity fields from state (caller may have passed hints via AgentInput)
    # or leave empty — they will be filled by the LLM from memory_context.
    return {
        "user_id": user_id,
        "session_id": session_id,
        "user_name": state.get("user_name", ""),
        "user_role": state.get("user_role", ""),
        "company_name": state.get("company_name", ""),
        "memory_context": memory_context,
        "retrieved_context": retrieved_chunks,
        "current_phase": "init",
        "plan_approved": False,
        "supervisor_retry_count": 0,
        "research_tasks": [],
        "research_findings": {},
        "report_sections": {},
        "report_draft": "",
        "final_report": "",
        "report_format": state.get("report_format", "markdown"),
        "research_plan": "",
        "supervisor_summary": "",
        "hitl_response_type": "",
    }


async def greeting_node(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Generates a personalised greeting using the loaded memory context.

    Acknowledges the user by name, references recent work if known,
    and asks a focused opening question to understand today's goal.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})

    llm = ChatOpenAI(
        model=cfg.model_name,
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
        temperature=0.3,
    )

    system_prompt = GREETING_SYSTEM_PROMPT.format(
        user_profile=memory.get("user_profile", ""),
        company_profile=memory.get("company_profile", ""),
        user_preferences=memory.get("user_preferences", ""),
        episodic_memory=memory.get("episodic_memory", ""),
    )

    greeting: AIMessage = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.get("query", "Hello")},
    ])

    return {
        "messages": [greeting],
        "current_phase": "planning",
    }
