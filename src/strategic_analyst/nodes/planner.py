"""
planner.py — Research planning node.

Input:  user query + retrieved_context + memory_context (+ message history for re-planning)
Output: structured ResearchPlan → stored as research_plan in state
Route:  always → hitl_plan_gate
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langgraph.types import Command

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import PLANNER_SYSTEM_PROMPT
from strategic_analyst.schemas import AgentState, ResearchPlan


async def planner_agent(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["hitl_plan_gate"]]:
    """
    Research planning node.

    Produces a structured ResearchPlan via structured output, then routes
    unconditionally to the HITL plan gate so the user can approve or revise.

    Re-planning (after user feedback at HITL gate) also passes through here —
    the full message history carries the feedback, so the LLM can adjust.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})
    context = state.get("retrieved_context", [])
    query = state.get("query", "")

    # Condense top-5 retrieved chunks into a single context block
    context_str = "\n\n".join(
        chunk.get("content", "") for chunk in context[:5] if chunk.get("content")
    ) or "No internal context retrieved yet."

    llm = ChatOpenAI(
        model=cfg.model_name,
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
        temperature=0.1,
    ).with_structured_output(ResearchPlan)

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        user_profile=memory.get("user_profile", ""),
        company_profile=memory.get("company_profile", ""),
        user_preferences=memory.get("user_preferences", ""),
        episodic_memory=memory.get("episodic_memory", ""),
        retrieved_context=context_str,
    )

    # Build messages: system + full conversation history (supports re-planning with feedback)
    messages: list = [{"role": "system", "content": system_prompt}]
    messages.extend(state.get("messages", []))

    # Ensure there is at least one human turn so the LLM has something to plan against
    has_human_turn = any(
        (m.get("role") == "user" if isinstance(m, dict) else getattr(m, "type", "") == "human")
        for m in messages[-6:]
    )
    if not has_human_turn:
        messages.append({"role": "user", "content": f"Please create a research plan for: {query}"})

    plan: ResearchPlan = await llm.ainvoke(messages)

    plan_markdown = _format_plan_as_markdown(plan)

    return Command(
        goto="hitl_plan_gate",
        update={
            "research_plan": plan_markdown,
            "research_tasks": [task.model_dump() for task in plan.tasks],
            "current_phase": "planning",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"I've prepared a research plan for your request:\n\n{plan_markdown}"
                    ),
                }
            ],
        },
    )


def _format_plan_as_markdown(plan: ResearchPlan) -> str:
    """Format a ResearchPlan Pydantic object as display-ready markdown."""
    lines = [
        f"# Research Plan: {plan.title}",
        f"\n**Objective:** {plan.objective}",
        f"\n**Background:** {plan.background}",
        f"\n**Expected Deliverable:** {plan.expected_deliverable}",
    ]
    if plan.frameworks:
        lines.append(f"\n**Frameworks:** {', '.join(plan.frameworks)}")

    lines.append("\n## Research Tasks\n")
    for task in plan.tasks:
        priority_icon = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(
            task.priority, "[?]"
        )
        lines.append(f"{priority_icon} **{task.task_id}**")
        lines.append(f"   Question: {task.question}")
        lines.append(f"   Sources: {', '.join(task.data_sources)}")
        if task.dependencies:
            lines.append(f"   Depends on: {', '.join(task.dependencies)}")
        lines.append("")

    return "\n".join(lines)
