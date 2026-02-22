"""
supervisor.py — Report subgraph supervisor nodes.

Three nodes live here:
  report_supervisor  — entry: plans sections, fans out via Send()
  section_reviewer   — post writer_agent: reviews each section, re-dispatches or continues
  report_assembler   — final: assembles all sections into report_draft
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import (
    REPORT_ASSEMBLER_PROMPT,
    REPORT_SUPERVISOR_SYSTEM_PROMPT,
    SECTION_REVIEW_PROMPT,
)
from strategic_analyst.schemas import (
    AgentState,
    ReportStructure,
    SupervisorReview,
)

_OVH_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
# Assembler uses main model (long-context not needed for final polish pass)
_ASSEMBLER_MODEL = "gpt-oss-20b"


def _ovh_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", _OVH_BASE),
        **kwargs,
    )


# ── Node 1: Report Supervisor (section planning + dispatch) ───────────────────

async def report_supervisor(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command:
    """
    Entry node of the report subgraph.

    Plans the report structure via LLM (ReportStructure), then dispatches each
    section to a writer_agent via Send() for parallel execution.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})
    findings = state.get("research_findings", {})
    supervisor_summary = state.get("supervisor_summary", "")
    research_plan = state.get("research_plan", "")
    user_id = state.get("user_id", cfg.user_id)

    structure_llm = _ovh_llm(
        model=cfg.model_name,
        temperature=0.1,
    ).with_structured_output(ReportStructure)

    structure: ReportStructure = await structure_llm.ainvoke([
        {
            "role": "system",
            "content": REPORT_SUPERVISOR_SYSTEM_PROMPT.format(
                user_profile=memory.get("user_profile", ""),
                user_preferences=memory.get("user_preferences", ""),
                company_profile=memory.get("company_profile", ""),
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research Plan:\n{research_plan}\n\n"
                f"Research Summary:\n{supervisor_summary}\n\n"
                "Plan the report structure and dispatch all sections."
            ),
        },
    ])

    section_sends = [
        Send(
            "writer_agent",
            {
                "section_id": section.section_id,
                "section_title": section.title,
                "section_instructions": (
                    section.instructions
                    + f"\nTarget approximately {section.word_count_target} words."
                ),
                "research_findings": findings,
                "supervisor_summary": supervisor_summary,
                "user_preferences": memory.get("user_preferences", ""),
                "user_profile": memory.get("user_profile", ""),
                "retry_count": 0,
                "supervisor_critique": "",
                "user_id": user_id,
                "messages": [],
            },
        )
        for section in structure.sections
    ]

    section_names = ", ".join(s.title for s in structure.sections)
    return Command(
        goto=section_sends,
        update={
            "current_phase": "reporting",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Starting report with {len(structure.sections)} sections: {section_names}. "
                        f"Global formatting: {structure.formatting_notes}"
                    ),
                }
            ],
        },
    )


# ── Node 2: Section Reviewer ──────────────────────────────────────────────────

async def section_reviewer(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["writer_agent", "report_assembler"]]:
    """
    Runs after all writer_agents finish (or after a retry round).

    Reviews each written section for quality and completeness.
    Re-dispatches rejected sections via Send(); proceeds to report_assembler
    when all sections are approved or max retries hit.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    sections = state.get("report_sections", {})
    retry_count = state.get("supervisor_retry_count", 0)
    findings = state.get("research_findings", {})
    memory = state.get("memory_context", {})
    supervisor_summary = state.get("supervisor_summary", "")
    user_id = state.get("user_id", cfg.user_id)

    if retry_count >= cfg.max_supervisor_retries:
        return Command(goto="report_assembler", update={})

    review_llm = _ovh_llm(
        model=cfg.utility_model_name,
        temperature=0.0,
    ).with_structured_output(SupervisorReview)

    re_dispatch: list[Send] = []

    for section_id, section_data in sections.items():
        review: SupervisorReview = await review_llm.ainvoke([
            {"role": "system", "content": SECTION_REVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Section title: {section_data.get('title', section_id)}\n"
                    f"Instructions were: {section_data.get('instructions', '')}\n\n"
                    f"Section content:\n{section_data.get('content', '')}"
                ),
            },
        ])

        if not review.approved:
            re_dispatch.append(
                Send(
                    "writer_agent",
                    {
                        "section_id": section_id,
                        "section_title": section_data.get("title", section_id),
                        "section_instructions": section_data.get("instructions", ""),
                        "research_findings": findings,
                        "supervisor_summary": supervisor_summary,
                        "user_preferences": memory.get("user_preferences", ""),
                        "user_profile": memory.get("user_profile", ""),
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

    return Command(goto="report_assembler", update={})


# ── Node 3: Report Assembler ──────────────────────────────────────────────────

async def report_assembler(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Combines all approved sections into a single polished report draft.
    Applies a final polish pass using the long-context model.
    Stores the result in report_draft.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    sections = state.get("report_sections", {})
    memory = state.get("memory_context", {})

    # Order sections by section_id (section_1, section_2, ...) for logical ordering
    def _sort_key(item: tuple) -> str:
        sid = item[0]
        # Extract trailing integer for proper numeric sort
        parts = sid.rsplit("_", 1)
        try:
            return f"{parts[0]}_{int(parts[-1]):04d}"
        except (ValueError, IndexError):
            return sid

    ordered_sections = sorted(sections.items(), key=_sort_key)

    combined = "\n\n---\n\n".join(
        f"## {data.get('title', sid)}\n\n{data.get('content', '')}"
        for sid, data in ordered_sections
    )

    if not combined.strip():
        return {
            "report_draft": "No sections were written.",
            "current_phase": "final",
            "messages": [{"role": "assistant", "content": "Report assembly failed — no sections available."}],
        }

    assembler_llm = _ovh_llm(
        model=_ASSEMBLER_MODEL,
        temperature=0.2,
    )

    polished = await assembler_llm.ainvoke([
        {
            "role": "system",
            "content": REPORT_ASSEMBLER_PROMPT.format(
                user_preferences=memory.get("user_preferences", ""),
                user_profile=memory.get("user_profile", ""),
            ),
        },
        {
            "role": "user",
            "content": (
                "Assemble the following sections into a polished, publication-ready report:\n\n"
                + combined
            ),
        },
    ])

    draft = polished.content if hasattr(polished, "content") else str(polished)

    return {
        "report_draft": draft,
        "current_phase": "final",
        "messages": [
            {
                "role": "assistant",
                "content": "Report draft assembled and ready for your review.",
            }
        ],
    }
