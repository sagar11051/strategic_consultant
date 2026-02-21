# Plan Step 08 — HITL Gates

## Goal
Implement all three human-in-the-loop interrupt nodes. These are the most critical UX touchpoints — they pause graph execution, present information to the user, and route based on their response. Every gate triggers memory updates.

---

## 8.1 HITL Framework

All gates use the Agent Inbox interrupt schema:

```python
request = {
    "action_request": {
        "action": "descriptive string",
        "args": {}
    },
    "config": {
        "allow_ignore": bool,
        "allow_respond": bool,
        "allow_edit": bool,
        "allow_accept": bool,
    },
    "description": "# Markdown shown to user"
}
response = interrupt([request])[0]
# response["type"] → "accept" | "edit" | "respond" | "ignore"
# response["args"] → depends on type
```

---

## 8.2 File: `src/strategic_analyst/nodes/hitl_gates.py`

### Gate 1 — Plan Review

```python
import os
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.nodes.memory_writer import trigger_memory_update


async def hitl_plan_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command[Literal["planner_agent", "research_subgraph", "__end__"]]:
    """
    HITL Gate 1 — Research Plan Review.

    Shows the user the proposed research plan.
    Routes:
    - accept → research_subgraph (proceed with current plan)
    - edit   → planner_agent (user edited the plan JSON; re-plan with edits)
    - respond → planner_agent (user gave textual feedback; re-plan)
    - ignore  → END

    Memory update: user's feedback informs user_preferences.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    plan = state.get("research_plan", "No plan generated.")
    tasks = state.get("research_tasks", [])

    task_count = len(tasks)
    sources_summary = set()
    for task in tasks:
        for src in task.get("data_sources", []):
            sources_summary.add(src)

    description = f"""# Research Plan for Your Review

{plan}

---
**{task_count} research tasks** will run in parallel.
**Data sources:** {', '.join(sources_summary) or 'company database + web'}

---
*Accept to proceed • Respond with feedback to modify • Edit to change specific tasks • Ignore to cancel*
"""

    request = {
        "action_request": {
            "action": "Review Research Plan",
            "args": {"research_tasks": tasks}  # Editable in Agent Inbox
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": description
    }

    # PAUSE GRAPH — wait for user response
    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        return Command(
            goto="research_subgraph",
            update={
                "plan_approved": True,
                "hitl_response_type": "accept",
                "messages": [{"role": "user", "content": "Plan approved. Starting research."}]
            }
        )

    elif response_type == "edit":
        # User edited the task list in Agent Inbox
        edited_args = response.get("args", {}).get("args", {})
        edited_tasks = edited_args.get("research_tasks", tasks)

        # Update memory: user cares about specific edits they made
        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[{
                "role": "user",
                "content": f"User edited the research plan tasks. Original: {tasks}. Edited to: {edited_tasks}."
            }],
            update_reason="User edited research plan. Update preferred research depth and scope.",
            utility_model=cfg.utility_model_name,
        )

        return Command(
            goto="planner_agent",
            update={
                "research_tasks": edited_tasks,
                "hitl_response_type": "edit",
                "messages": [{
                    "role": "user",
                    "content": f"I've modified the research tasks. Please revise the plan accordingly."
                }]
            }
        )

    elif response_type == "respond":
        # User provided textual feedback
        feedback = response.get("args", "")

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[{
                "role": "user",
                "content": f"User gave this feedback on the research plan: {feedback}"
            }],
            update_reason="User gave feedback on research plan. Update research style preferences.",
            utility_model=cfg.utility_model_name,
        )

        return Command(
            goto="planner_agent",
            update={
                "hitl_response_type": "respond",
                "messages": [{"role": "user", "content": str(feedback)}]
            }
        )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Research cancelled."}]
            }
        )
```

---

### Gate 2 — Discovery Review

```python
async def hitl_discovery_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command[Literal["research_subgraph", "report_subgraph", "__end__"]]:
    """
    HITL Gate 2 — Research Discovery Review.

    The research supervisor presents key discoveries and asks intelligent follow-up questions.
    Routes:
    - accept  → report_subgraph (user satisfied, generate report)
    - respond → research_subgraph (user wants to go deeper on something)
    - ignore  → END

    Memory update:
    - User responses update user_preferences (what they found interesting)
    - Key discoveries noted in episodic_memory (already done by discovery_synthesiser)
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    summary = state.get("supervisor_summary", "No research summary available.")
    findings = state.get("research_findings", {})

    # Build rich display
    findings_display = "\n\n".join(
        f"**{tid}:** {data.get('answer', '')[:300]}..."
        for tid, data in list(findings.items())[:5]
    )

    description = f"""# Research Discoveries

## Summary
{summary}

## Key Findings
{findings_display}

---
*Accept to proceed to report writing*
*Respond with follow-up direction (e.g. "Go deeper on competitor X's pricing strategy")*
*Ignore to end the session without a report*
"""

    request = {
        "action_request": {
            "action": "Review Research Discoveries",
            "args": {}
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": True,
        },
        "description": description
    }

    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        return Command(
            goto="report_subgraph",
            update={
                "hitl_response_type": "accept",
                "current_phase": "reporting",
                "messages": [{"role": "user", "content": "Research approved. Please write the report."}]
            }
        )

    elif response_type == "respond":
        follow_up = response.get("args", "")

        # Record user's interest in specific discovery areas
        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[{
                "role": "user",
                "content": f"User asked for deeper research on: {follow_up}"
            }],
            update_reason="User requested deeper research on specific findings. Update research depth preferences.",
            utility_model=cfg.utility_model_name,
        )

        # Add new query to messages for the research supervisor to pick up
        return Command(
            goto="research_subgraph",
            update={
                "hitl_response_type": "respond",
                "supervisor_retry_count": 0,  # Reset retry count for new research pass
                "messages": [{
                    "role": "user",
                    "content": f"Please investigate further: {follow_up}"
                }]
            }
        )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Session ended. No report generated."}]
            }
        )
```

---

### Gate 3 — Final Report Review

```python
async def hitl_final_gate(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command[Literal["save_report_node", "research_subgraph", "planner_agent", "__end__"]]:
    """
    HITL Gate 3 — Final Report Review. Most flexible gate.

    Shows the report draft. User can:
    - accept         → choose format → save_report_node
    - respond        → ask questions (answered by returning to research context) OR
                       request format changes → stays in final review loop
    - edit           → edit specific sections → re-assemble
    - ignore         → end without saving

    Special respond commands:
    - "re-research: <topic>"  → research_subgraph
    - "re-plan: <direction>"  → planner_agent
    - "format: markdown|pdf|json" → update report_format, stay in loop
    - anything else           → Q&A mode (research supervisor answers inline)
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})
    draft = state.get("report_draft", "")
    report_format = state.get("report_format", "markdown")

    description = f"""# Report Draft — Ready for Review

**Format:** {report_format}

---

{draft[:3000]}{'...' if len(draft) > 3000 else ''}

---

**Commands:**
- *Accept* → Save report in {report_format} format
- *Respond* with any question → I'll answer from the research context
- *Respond* "re-research: [topic]" → Conduct additional research
- *Respond* "re-plan: [direction]" → Start fresh with new direction
- *Respond* "format: markdown/pdf/json" → Change output format
- *Edit* → Modify specific sections
- *Ignore* → End without saving
"""

    request = {
        "action_request": {
            "action": "Review Final Report",
            "args": {"report_sections": state.get("report_sections", {})}
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": description
    }

    response = interrupt([request])[0]
    response_type = response.get("type", "ignore")

    if response_type == "accept":
        # Update memory with user's approval context
        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[{
                "role": "user",
                "content": f"User approved the report in {report_format} format."
            }],
            update_reason="User approved report. Confirm preferred format and style.",
            utility_model=cfg.utility_model_name,
        )

        return Command(
            goto="save_report_node",
            update={
                "hitl_response_type": "accept",
                "final_report": draft,
                "current_phase": "final",
                "messages": [{"role": "user", "content": "Report approved. Please save it."}]
            }
        )

    elif response_type == "edit":
        # User edited specific section(s) in Agent Inbox
        edited_sections = response.get("args", {}).get("args", {}).get("report_sections", {})

        await trigger_memory_update(
            store=store,
            user_id=cfg.user_id,
            namespace_type="user_preferences",
            context_messages=[{
                "role": "user",
                "content": f"User edited report sections. Changes reflect style preferences."
            }],
            update_reason="User edited report sections. Update style and formatting preferences.",
            utility_model=cfg.utility_model_name,
        )

        # Merge edits and re-assemble
        current_sections = state.get("report_sections", {})
        merged_sections = {**current_sections, **edited_sections}

        return Command(
            goto="report_subgraph",  # Re-run assembler with edits
            update={
                "report_sections": merged_sections,
                "hitl_response_type": "edit",
                "messages": [{"role": "user", "content": "I've edited some sections. Please re-assemble."}]
            }
        )

    elif response_type == "respond":
        feedback = str(response.get("args", "")).strip().lower()

        # Parse special commands
        if feedback.startswith("re-research:"):
            topic = feedback.replace("re-research:", "").strip()
            return Command(
                goto="research_subgraph",
                update={
                    "hitl_response_type": "respond",
                    "supervisor_retry_count": 0,
                    "messages": [{"role": "user", "content": f"Please research further: {topic}"}]
                }
            )

        elif feedback.startswith("re-plan:"):
            direction = feedback.replace("re-plan:", "").strip()
            return Command(
                goto="planner_agent",
                update={
                    "hitl_response_type": "respond",
                    "messages": [{"role": "user", "content": f"Start fresh with new direction: {direction}"}]
                }
            )

        elif feedback.startswith("format:"):
            fmt = feedback.replace("format:", "").strip()
            if fmt in ["markdown", "pdf", "json"]:
                return Command(
                    goto="hitl_final_gate",  # Loop back to this gate with new format
                    update={
                        "report_format": fmt,
                        "hitl_response_type": "respond",
                        "messages": [{
                            "role": "assistant",
                            "content": f"Format updated to {fmt}. Please review again."
                        }]
                    }
                )

        else:
            # Q&A mode: answer from research context, loop back
            # Route to research supervisor for inline Q&A (it has the most context)
            return Command(
                goto="research_subgraph",
                update={
                    "hitl_response_type": "respond",
                    "messages": [{
                        "role": "user",
                        "content": str(response.get("args", ""))
                    }]
                }
            )

    else:  # ignore
        return Command(
            goto="__end__",
            update={
                "hitl_response_type": "ignore",
                "current_phase": "end",
                "messages": [{"role": "assistant", "content": "Session ended without saving."}]
            }
        )
```

---

## 8.3 Memory Update Summary Per Gate

| Gate | Response Type | Memory Updated | Reason |
|---|---|---|---|
| Gate 1 (Plan) | edit | user_preferences | User modified plan → scope/depth preferences |
| Gate 1 (Plan) | respond | user_preferences | User text feedback → research style |
| Gate 2 (Discovery) | respond | user_preferences | User follow-up interests |
| Gate 3 (Report) | accept | user_preferences | Confirms format preference |
| Gate 3 (Report) | edit | user_preferences | Section edits → style preferences |
| Discovery Synthesiser | auto | episodic_memory | Key findings with date |
| Save Report | auto | episodic_memory | Session summary with date |

---

## 8.4 Resuming After Interrupt

From the client/notebook:

```python
# Trigger interrupt (graph pauses here)
for chunk in graph.stream({"user_id": "u1", "query": "..."}, config=thread_config):
    print(chunk)

# Inspect state while paused
state = graph.get_state(thread_config)
print(state.next)  # → ['hitl_plan_gate']

# Resume with user's response
for chunk in graph.stream(
    Command(resume=[{"type": "accept", "args": ""}]),
    config=thread_config
):
    print(chunk)

# Resume with textual feedback
for chunk in graph.stream(
    Command(resume=[{
        "type": "respond",
        "args": "Focus more on the APAC market segment"
    }]),
    config=thread_config
):
    print(chunk)
```

---

## Completion Checklist

- [ ] `hitl_plan_gate` written with all 4 response types
- [ ] `hitl_discovery_gate` written with all 3 response types
- [ ] `hitl_final_gate` written with all special command parsing
- [ ] All gates trigger appropriate memory updates
- [ ] Each gate has correct `Command[Literal[...]]` return type hint
- [ ] Gates tested manually: interrupt fires → resume → routing correct
- [ ] Special commands in Gate 3 tested (re-research:, re-plan:, format:)
