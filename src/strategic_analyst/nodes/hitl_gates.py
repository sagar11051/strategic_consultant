"""
hitl_gates.py — Three human-in-the-loop interrupt nodes.

Each gate pauses graph execution via interrupt(), presents information to the
user in the Agent Inbox format, then routes based on the response type.

Gate 1 — hitl_plan_gate
  Shows: research_plan + task count
  accept  → research_subgraph
  edit    → planner_agent (with edited tasks)
  respond → planner_agent (with textual feedback)
  ignore  → END

Gate 2 — hitl_discovery_gate
  Shows: supervisor_summary + key findings
  accept  → report_subgraph
  respond → research_subgraph (additional research direction)
  ignore  → END

Gate 3 — hitl_final_gate
  Shows: report_draft preview
  accept                       → save_report_node
  edit                         → report_subgraph (re-assemble with edits)
  respond "re-research: <topic>" → research_subgraph
  respond "re-plan: <direction>" → planner_agent
  respond "format: markdown|pdf|json" → loops back with new format
  respond (other)              → research_subgraph (Q&A pass)
  ignore                       → END

All gates write to user_preferences memory on relevant feedback to preserve
the user's research and formatting preferences across sessions.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.nodes.memory_writer import trigger_memory_update
from strategic_analyst.schemas import AgentState


# ── Gate 1: Research Plan Review ──────────────────────────────────────────────

async def hitl_plan_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["planner_agent", "research_subgraph", "__end__"]]:
    """
    HITL Gate 1 — Research Plan Review.

    Presents the proposed research plan to the user.
    Pauses execution until the user responds via the Agent Inbox.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    plan = state.get("research_plan", "No plan generated.")
    tasks = state.get("research_tasks", [])

    sources_summary = set()
    for task in tasks:
        for src in task.get("data_sources", []):
            sources_summary.add(src)

    description = (
        f"# Research Plan for Your Review\n\n"
        f"{plan}\n\n"
        f"---\n"
        f"**{len(tasks)} research tasks** will run in parallel.\n"
        f"**Data sources:** {', '.join(sources_summary) or 'company database + web'}\n\n"
        f"---\n"
        f"*Accept to proceed • Respond with feedback to modify • "
        f"Edit to change specific tasks • Ignore to cancel*"
    )

    request = {
        "action_request": {
            "action": "Review Research Plan",
            "args": {"research_tasks": tasks},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": description,
    }

    # ── PAUSE — wait for user ─────────────────────────────────────────────────
    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        return Command(
            goto="research_subgraph",
            update={
                "plan_approved": True,
                "hitl_response_type": "accept",
                "supervisor_retry_count": 0,
                "messages": [{"role": "user", "content": "Plan approved. Starting research."}],
            },
        )

    elif response_type == "edit":
        edited_args = response.get("args", {})
        # Agent Inbox wraps edits under args.args
        if isinstance(edited_args, dict):
            edited_tasks = edited_args.get("args", {}).get("research_tasks", tasks)
        else:
            edited_tasks = tasks

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[
                {
                    "role": "user",
                    "content": (
                        f"User edited the research plan tasks. "
                        f"Original task IDs: {[t.get('task_id') for t in tasks]}. "
                        f"Edited to: {[t.get('task_id') for t in edited_tasks]}."
                    ),
                }
            ],
            update_reason="User edited research plan. Update preferred research scope and depth.",
            model_name=cfg.utility_model_name,
        )

        return Command(
            goto="planner_agent",
            update={
                "research_tasks": edited_tasks,
                "hitl_response_type": "edit",
                "messages": [
                    {
                        "role": "user",
                        "content": "I've modified the research tasks. Please revise the plan accordingly.",
                    }
                ],
            },
        )

    elif response_type == "respond":
        feedback = str(response.get("args", "")).strip()

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[
                {
                    "role": "user",
                    "content": f"User gave this feedback on the research plan: {feedback}",
                }
            ],
            update_reason="User gave feedback on research plan. Update research style preferences.",
            model_name=cfg.utility_model_name,
        )

        return Command(
            goto="planner_agent",
            update={
                "hitl_response_type": "respond",
                "messages": [{"role": "user", "content": feedback}],
            },
        )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Research cancelled by user."}],
            },
        )


# ── Gate 2: Discovery Review ──────────────────────────────────────────────────

async def hitl_discovery_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["research_subgraph", "report_subgraph", "__end__"]]:
    """
    HITL Gate 2 — Research Discovery Review.

    Shows the supervisor synthesis and key findings.
    User can approve (→ report), ask for deeper research (→ research), or end.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    summary = state.get("supervisor_summary", "No research summary available.")
    findings = state.get("research_findings", {})

    findings_display = "\n\n".join(
        f"**{tid}:** {str(data.get('answer', ''))[:400]}..."
        for tid, data in list(findings.items())[:5]
    )

    description = (
        f"# Research Discoveries\n\n"
        f"## Summary\n{summary}\n\n"
        f"## Key Findings (preview)\n{findings_display}\n\n"
        f"---\n"
        f"*Accept to proceed to report writing*\n"
        f"*Respond with follow-up direction (e.g. \"Go deeper on competitor X's pricing\")*\n"
        f"*Ignore to end the session without a report*"
    )

    request = {
        "action_request": {
            "action": "Review Research Discoveries",
            "args": {},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": True,
        },
        "description": description,
    }

    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        return Command(
            goto="report_subgraph",
            update={
                "hitl_response_type": "accept",
                "current_phase": "reporting",
                "supervisor_retry_count": 0,
                "messages": [
                    {"role": "user", "content": "Research approved. Please write the report."}
                ],
            },
        )

    elif response_type == "respond":
        follow_up = str(response.get("args", "")).strip()

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[
                {
                    "role": "user",
                    "content": f"User asked for deeper research on: {follow_up}",
                }
            ],
            update_reason="User requested deeper research on specific findings. Update depth preferences.",
            model_name=cfg.utility_model_name,
        )

        return Command(
            goto="research_subgraph",
            update={
                "hitl_response_type": "respond",
                "supervisor_retry_count": 0,
                "messages": [{"role": "user", "content": f"Please investigate further: {follow_up}"}],
            },
        )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Session ended. No report generated."}],
            },
        )


# ── Gate 3: Final Report Review ───────────────────────────────────────────────

async def hitl_final_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal[
    "save_report_node", "report_subgraph", "research_subgraph",
    "planner_agent", "hitl_final_gate", "__end__"
]]:
    """
    HITL Gate 3 — Final Report Review.

    Shows the report draft. Supports special respond commands:
      "re-research: <topic>"          → research_subgraph
      "re-plan: <direction>"          → planner_agent
      "format: markdown|pdf|json"     → update report_format, loop back
      (anything else)                 → research_subgraph (Q&A pass)

    accept → save_report_node
    edit   → report_subgraph (re-assemble with edited sections)
    ignore → END
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    draft = state.get("report_draft", "")
    report_format = state.get("report_format", "markdown")

    draft_preview = draft[:3000] + ("..." if len(draft) > 3000 else "")

    description = (
        f"# Report Draft — Ready for Review\n\n"
        f"**Format:** {report_format}\n\n"
        f"---\n\n"
        f"{draft_preview}\n\n"
        f"---\n\n"
        f"**Commands:**\n"
        f"- *Accept* → Save report in {report_format} format\n"
        f"- *Respond* with any question → Answered from research context\n"
        f"- *Respond* `re-research: [topic]` → Conduct additional research\n"
        f"- *Respond* `re-plan: [direction]` → Start fresh with new direction\n"
        f"- *Respond* `format: markdown/pdf/json` → Change output format\n"
        f"- *Edit* → Modify specific sections then re-assemble\n"
        f"- *Ignore* → End without saving"
    )

    request = {
        "action_request": {
            "action": "Review Final Report",
            "args": {"report_sections": state.get("report_sections", {})},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": description,
    }

    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[
                {
                    "role": "user",
                    "content": f"User approved the report in {report_format} format.",
                }
            ],
            update_reason="User approved report. Confirm preferred format and style.",
            model_name=cfg.utility_model_name,
        )

        return Command(
            goto="save_report_node",
            update={
                "hitl_response_type": "accept",
                "final_report": draft,
                "current_phase": "final",
                "messages": [{"role": "user", "content": "Report approved. Please save it."}],
            },
        )

    elif response_type == "edit":
        edited_args = response.get("args", {})
        if isinstance(edited_args, dict):
            edited_sections = edited_args.get("args", {}).get("report_sections", {})
        else:
            edited_sections = {}

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[
                {
                    "role": "user",
                    "content": "User edited report sections — reflects style and content preferences.",
                }
            ],
            update_reason="User edited report sections. Update style and formatting preferences.",
            model_name=cfg.utility_model_name,
        )

        current_sections = state.get("report_sections", {})
        merged_sections = {**current_sections, **edited_sections}

        return Command(
            goto="report_subgraph",
            update={
                "report_sections": merged_sections,
                "hitl_response_type": "edit",
                "supervisor_retry_count": 0,
                "messages": [
                    {"role": "user", "content": "I've edited some sections. Please re-assemble the report."}
                ],
            },
        )

    elif response_type == "respond":
        feedback = str(response.get("args", "")).strip()
        feedback_lower = feedback.lower()

        if feedback_lower.startswith("re-research:"):
            topic = feedback[len("re-research:"):].strip()
            return Command(
                goto="research_subgraph",
                update={
                    "hitl_response_type": "respond",
                    "supervisor_retry_count": 0,
                    "messages": [{"role": "user", "content": f"Please research further: {topic}"}],
                },
            )

        elif feedback_lower.startswith("re-plan:"):
            direction = feedback[len("re-plan:"):].strip()
            return Command(
                goto="planner_agent",
                update={
                    "hitl_response_type": "respond",
                    "messages": [{"role": "user", "content": f"Start fresh with new direction: {direction}"}],
                },
            )

        elif feedback_lower.startswith("format:"):
            fmt = feedback_lower[len("format:"):].strip()
            if fmt in ("markdown", "pdf", "json"):
                return Command(
                    goto="hitl_final_gate",
                    update={
                        "report_format": fmt,
                        "hitl_response_type": "respond",
                        "messages": [
                            {"role": "assistant", "content": f"Format updated to {fmt}. Please review again."}
                        ],
                    },
                )
            # Unknown format — fall through to Q&A

        # Q&A mode or unrecognised respond command → route to research for inline answer
        return Command(
            goto="research_subgraph",
            update={
                "hitl_response_type": "respond",
                "supervisor_retry_count": 0,
                "messages": [{"role": "user", "content": feedback}],
            },
        )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Session ended without saving."}],
            },
        )
