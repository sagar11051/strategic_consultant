"""
supervisor.py — Research subgraph supervisor nodes.

Three nodes live here:
  research_supervisor    — entry: reads research_tasks, fans out via Send()
  supervisor_review      — post task_agent: reviews each finding, re-dispatches or continues
  discovery_synthesiser  — final: synthesises all findings → supervisor_summary
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.nodes.memory_writer import trigger_memory_update
from strategic_analyst.prompts import (
    DISCOVERY_SYNTHESIS_PROMPT,
    SUPERVISOR_REVIEW_PROMPT,
)
from strategic_analyst.schemas import AgentState, SupervisorDiscoveries, SupervisorReview

_OVH_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


def _ovh_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", _OVH_BASE),
        **kwargs,
    )


# ── Node 1: Research Supervisor (dispatch) ────────────────────────────────────

async def research_supervisor(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command:
    """
    Entry node of the research subgraph.

    Reads research_tasks from state (structured by the planner) and fans them
    out in parallel via Send(), one Send per task.

    No LLM call here — the task list is already structured. The supervisor's
    role at this stage is allocation, not reasoning.
    """
    tasks = state.get("research_tasks", [])
    context_chunks = state.get("retrieved_context", [])
    user_id = state.get("user_id", "")

    # Give each task agent a slice of the retrieved context as a head-start
    context_snippet = "\n\n".join(
        chunk.get("content", "")
        for chunk in context_chunks[:3]
        if chunk.get("content")
    )

    task_sends = [
        Send(
            "task_agent",
            {
                "task_id": task["task_id"],
                "question": task["question"],
                "data_sources": task.get("data_sources", ["company_db", "web"]),
                "context": context_snippet,
                "retry_count": 0,
                "supervisor_critique": "",
                "user_id": user_id,
                "messages": [],
            },
        )
        for task in tasks
    ]

    return Command(
        goto=task_sends,
        update={
            "current_phase": "research",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Dispatching {len(tasks)} research task(s) in parallel: "
                        + ", ".join(t["task_id"] for t in tasks)
                    ),
                }
            ],
        },
    )


# ── Node 2: Supervisor Review ─────────────────────────────────────────────────

async def supervisor_review(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["task_agent", "discovery_synthesiser"]]:
    """
    Runs after all task_agents finish (or after a retry round).

    For each task:
      - If a finding exists → LLM reviews it (SupervisorReview)
        - approved  → keep finding
        - rejected  → re-dispatch via Send("task_agent", ...)
      - If no finding → re-dispatch immediately

    When all tasks are approved (or max retries hit) → goto discovery_synthesiser.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    findings = state.get("research_findings", {})
    tasks = state.get("research_tasks", [])
    retry_count = state.get("supervisor_retry_count", 0)
    user_id = state.get("user_id", cfg.user_id)

    # If max retries hit, accept whatever we have
    if retry_count >= cfg.max_supervisor_retries:
        return Command(goto="discovery_synthesiser", update={})

    review_llm = _ovh_llm(
        model=cfg.model_name,
        temperature=0.0,
    ).with_structured_output(SupervisorReview)

    re_dispatch: list[Send] = []

    for task in tasks:
        task_id = task["task_id"]
        finding = findings.get(task_id)

        if finding is None:
            # Task agent never returned — re-dispatch
            re_dispatch.append(
                Send(
                    "task_agent",
                    {
                        "task_id": task_id,
                        "question": task["question"],
                        "data_sources": task.get("data_sources", ["company_db", "web"]),
                        "context": "",
                        "retry_count": retry_count + 1,
                        "supervisor_critique": "No findings were returned. Please retry the research task.",
                        "user_id": user_id,
                        "messages": [],
                    },
                )
            )
            continue

        # Review the finding
        review: SupervisorReview = await review_llm.ainvoke([
            {"role": "system", "content": SUPERVISOR_REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task question: {task['question']}\n\n"
                    f"Finding:\n{json.dumps(finding, indent=2, default=str)}"
                ),
            },
        ])

        if not review.approved:
            re_dispatch.append(
                Send(
                    "task_agent",
                    {
                        "task_id": task_id,
                        "question": review.follow_up_question or task["question"],
                        "data_sources": task.get("data_sources", ["company_db", "web"]),
                        "context": "",
                        "retry_count": retry_count + 1,
                        "supervisor_critique": review.critique,
                        "user_id": user_id,
                        "messages": [],
                    },
                )
            )

    if re_dispatch:
        return Command(
            goto=re_dispatch,
            update={"supervisor_retry_count": retry_count + 1},
        )

    return Command(goto="discovery_synthesiser", update={})


# ── Node 3: Discovery Synthesiser ─────────────────────────────────────────────

async def discovery_synthesiser(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Synthesises all approved research findings into a SupervisorDiscoveries object.
    Stores the summary in state and writes key discoveries to episodic memory.
    Routes to END (returns to main graph where hitl_discovery_gate awaits).
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    findings = state.get("research_findings", {})
    memory = state.get("memory_context", {})
    user_id = state.get("user_id", cfg.user_id)

    findings_text = "\n\n".join(
        f"**{task_id}:** {data.get('answer', '')}"
        for task_id, data in findings.items()
    ) or "No research findings available."

    synthesis_llm = _ovh_llm(
        model=cfg.model_name,
        temperature=0.2,
    ).with_structured_output(SupervisorDiscoveries)

    discoveries: SupervisorDiscoveries = await synthesis_llm.ainvoke([
        {
            "role": "system",
            "content": DISCOVERY_SYNTHESIS_PROMPT.format(
                user_profile=memory.get("user_profile", ""),
                user_preferences=memory.get("user_preferences", ""),
            ),
        },
        {
            "role": "user",
            "content": f"Research findings to synthesise:\n\n{findings_text}",
        },
    ])

    # Write key discoveries to episodic memory
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    discoveries_bullet = "\n".join(f"- {d}" for d in discoveries.key_discoveries)
    await trigger_memory_update(
        store=store,
        user_id=user_id,
        namespace_type="episodic_memory",
        context_messages=[
            {
                "role": "user",
                "content": (
                    f"On {today}, completed a research phase. Key discoveries:\n"
                    f"{discoveries_bullet}"
                ),
            }
        ],
        update_reason="Research phase completed. Record key discoveries as temporal notes.",
        model_name=cfg.utility_model_name,
    )

    return {
        "supervisor_summary": discoveries.summary,
        "current_phase": "discoveries",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"**Research Complete**\n\n{discoveries.summary}\n\n"
                    f"**Key Discoveries:**\n"
                    + "\n".join(f"- {d}" for d in discoveries.key_discoveries)
                ),
            }
        ],
    }
