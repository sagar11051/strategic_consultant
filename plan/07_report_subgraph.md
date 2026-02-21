# Plan Step 07 — Report Writing Subgraph

## Goal
Build the report writing subgraph. A report supervisor decomposes the findings into sections, dispatches writer agents (parallel via `Send()`), reviews each section, then assembles the final report in the user's preferred style.

---

## 7.1 Subgraph Structure

```
report_graph (StateGraph)
       │
       ▼
[report_supervisor]
   ├── Reads: research_findings, supervisor_summary, memory_context
   ├── Plans report structure (sections)
   └── Dispatches via Send() → parallel writer agents
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
[writer_agent_1] [writer_agent_2] [writer_agent_N]
  (exec summary)  (market analysis) (recommendations)
    │        │        │
    └────────┴────────┘
             │ all sections return
             ▼
[section_reviewer]
   ├── Approve or re-write each section
   └── (re-dispatch bad sections via Send())
             │
             ▼
[report_assembler]
   ├── Combines all sections into full report
   ├── Applies user formatting preferences
   └── → END (returns assembled report to main graph)
```

---

## 7.2 Section Writer State

```python
# In schemas.py — add this
class WriterAgentState(TypedDict):
    """State passed to each writer agent via Send()."""
    section_id: str
    section_title: str
    section_instructions: str     # what this section must cover
    research_findings: dict       # full findings for reference
    supervisor_summary: str
    user_preferences: str         # style/format preferences from memory
    user_profile: str
    retry_count: int
    supervisor_critique: str
    user_id: str
    messages: list
```

---

## 7.3 File: `src/strategic_analyst/subgraphs/report/supervisor.py`

### Report Supervisor — Section Planning & Dispatch

```python
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import REPORT_SUPERVISOR_SYSTEM_PROMPT


class ReportSection(BaseModel):
    section_id: str
    title: str
    instructions: str = Field(description="What this section must cover, what data to use, tone")
    word_count_target: int = Field(default=400)


class ReportStructure(BaseModel):
    """Planned structure of the report."""
    sections: list[ReportSection]
    formatting_notes: str = Field(description="Global formatting instructions based on user preferences")


async def report_supervisor(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command:
    """
    Report supervisor — plans the report structure and dispatches writer agents.

    Reads the user's preference for frameworks and report style from memory.
    Creates a tailored section plan and dispatches each section in parallel.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})
    findings = state.get("research_findings", {})
    summary = state.get("supervisor_summary", "")
    research_plan = state.get("research_plan", "")

    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    ).with_structured_output(ReportStructure)

    structure: ReportStructure = await llm.ainvoke([
        {"role": "system", "content": REPORT_SUPERVISOR_SYSTEM_PROMPT.format(
            user_profile=memory.get("user_profile", ""),
            user_preferences=memory.get("user_preferences", ""),
            company_profile=memory.get("company_profile", ""),
        )},
        {"role": "user", "content": (
            f"Research Plan:\n{research_plan}\n\n"
            f"Research Summary:\n{summary}\n\n"
            "Please plan the report structure."
        )}
    ])

    # Dispatch section writers in parallel
    section_sends = [
        Send("writer_agent", {
            "section_id": section.section_id,
            "section_title": section.title,
            "section_instructions": section.instructions + f"\nTarget ~{section.word_count_target} words.",
            "research_findings": findings,
            "supervisor_summary": summary,
            "user_preferences": memory.get("user_preferences", ""),
            "user_profile": memory.get("user_profile", ""),
            "retry_count": 0,
            "supervisor_critique": "",
            "user_id": state["user_id"],
            "messages": [],
        })
        for section in structure.sections
    ]

    return Command(
        goto=section_sends,
        update={
            "current_phase": "reporting",
            "messages": [{
                "role": "assistant",
                "content": (
                    f"Starting report with {len(structure.sections)} sections: "
                    + ", ".join(s.title for s in structure.sections)
                )
            }]
        }
    )
```

### Report Supervisor — Section Review Node

```python
from strategic_analyst.schemas import SupervisorReview as SectionReview


async def section_reviewer(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command:
    """
    Reviews each written section.
    Approved → accumulate in report_sections.
    Rejected → re-dispatch writer via Send().
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    sections = state.get("report_sections", {})
    retry_count = state.get("supervisor_retry_count", 0)

    if retry_count >= cfg.max_supervisor_retries:
        return Command(goto="report_assembler", update={})

    llm = ChatAnthropic(
        model=cfg.utility_model_name,  # Haiku for efficiency
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0
    ).with_structured_output(SectionReview)

    re_dispatch = []
    for section_id, section_data in sections.items():
        review: SectionReview = await llm.ainvoke([
            {"role": "system", "content": SECTION_REVIEW_PROMPT},
            {"role": "user", "content": (
                f"Section: {section_data.get('title', section_id)}\n\n"
                f"Content:\n{section_data.get('content', '')}\n\n"
                f"Instructions were: {section_data.get('instructions', '')}"
            )}
        ])

        if not review.approved:
            re_dispatch.append(
                Send("writer_agent", {
                    "section_id": section_id,
                    "section_title": section_data.get("title", section_id),
                    "section_instructions": section_data.get("instructions", ""),
                    "research_findings": state.get("research_findings", {}),
                    "supervisor_summary": state.get("supervisor_summary", ""),
                    "user_preferences": state.get("memory_context", {}).get("user_preferences", ""),
                    "user_profile": state.get("memory_context", {}).get("user_profile", ""),
                    "retry_count": retry_count + 1,
                    "supervisor_critique": review.critique,
                    "user_id": state["user_id"],
                    "messages": [],
                })
            )

    if re_dispatch:
        return Command(goto=re_dispatch, update={"supervisor_retry_count": retry_count + 1})
    else:
        return Command(goto="report_assembler", update={})
```

---

## 7.4 File: `src/strategic_analyst/subgraphs/report/writer_agent.py`

```python
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from strategic_analyst.schemas import WriterAgentState
from strategic_analyst.tools.base import get_tools, make_tool_node, REPORT_WRITER_TOOLS
from strategic_analyst.prompts import WRITER_AGENT_SYSTEM_PROMPT


def build_writer_agent_graph(store: BaseStore, user_id: str):
    """
    Factory: builds a writer agent mini graph.
    Writer agent uses RAG and memory tools to gather additional context,
    then writes the assigned section.

    Flow: START → [writer_llm] → (tool?) → [writer_tool_node] → [writer_llm] → END
    """
    tools = get_tools(REPORT_WRITER_TOOLS)
    llm = ChatAnthropic(
        model=os.getenv("WRITER_MODEL", "claude-sonnet-4-6"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3  # Slightly higher for writing quality
    ).bind_tools(tools, tool_choice="any")  # tools available but not forced

    async def writer_llm_call(state: WriterAgentState, config: RunnableConfig):
        """LLM call for a writer agent."""
        critique = state.get("supervisor_critique", "")
        critique_section = f"\n\n**Revision Required:** {critique}" if critique else ""

        system = WRITER_AGENT_SYSTEM_PROMPT.format(
            section_title=state["section_title"],
            section_instructions=state["section_instructions"],
            user_profile=state.get("user_profile", ""),
            user_preferences=state.get("user_preferences", ""),
            supervisor_summary=state.get("supervisor_summary", ""),
            critique_section=critique_section,
        )

        # Provide research findings as user context
        findings_text = "\n\n".join(
            f"**{tid}:** {data.get('answer', '')}"
            for tid, data in state.get("research_findings", {}).items()
        )

        messages = [{"role": "system", "content": system}] + state.get("messages", [])
        if not state.get("messages"):
            messages.append({
                "role": "user",
                "content": f"Research findings for context:\n\n{findings_text}\n\nNow write the section."
            })

        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    async def writer_tool_node(state: WriterAgentState, config: RunnableConfig):
        tool_fn = make_tool_node(tools, store, state["user_id"])
        return await tool_fn(state)

    def should_continue_writer(state: WriterAgentState):
        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            if any(tc["name"] == "Done" for tc in last.tool_calls):
                return END
            return "writer_tool_node"
        return END

    builder = StateGraph(WriterAgentState)
    builder.add_node("writer_llm_call", writer_llm_call)
    builder.add_node("writer_tool_node", writer_tool_node)
    builder.add_edge(START, "writer_llm_call")
    builder.add_conditional_edges("writer_llm_call", should_continue_writer, {
        "writer_tool_node": "writer_tool_node",
        END: END,
    })
    builder.add_edge("writer_tool_node", "writer_llm_call")

    return builder.compile()  # No checkpointer
```

---

## 7.5 Report Assembler Node

```python
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import REPORT_ASSEMBLER_PROMPT


async def report_assembler(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """
    Combines all approved sections into the final report.
    Applies global formatting based on user preferences.
    Produces a polished, cohesive document.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    sections = state.get("report_sections", {})
    memory = state.get("memory_context", {})
    preferences = memory.get("user_preferences", "")

    # Order sections by section_id (task_1 first, etc.)
    ordered_sections = sorted(sections.items(), key=lambda x: x[0])
    combined = "\n\n---\n\n".join(
        f"## {data.get('title', sid)}\n\n{data.get('content', '')}"
        for sid, data in ordered_sections
    )

    # Final polish pass via LLM
    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.2
    )

    polished = await llm.ainvoke([
        {"role": "system", "content": REPORT_ASSEMBLER_PROMPT.format(
            user_preferences=preferences,
            user_profile=memory.get("user_profile", ""),
        )},
        {"role": "user", "content": f"Assemble this report into a polished final document:\n\n{combined}"}
    ])

    return {
        "report_draft": polished.content,
        "current_phase": "final",
        "messages": [{
            "role": "assistant",
            "content": "Report draft assembled. Ready for your review."
        }]
    }
```

---

## 7.6 File: `src/strategic_analyst/subgraphs/report/report_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from strategic_analyst.schemas import AgentState
from langgraph.store.base import BaseStore

from .supervisor import report_supervisor, section_reviewer
from .writer_agent import build_writer_agent_graph
from .assembler import report_assembler


def build_report_subgraph(store: BaseStore, user_id: str):
    """Factory: build and compile the report writing subgraph."""
    writer_agent_graph = build_writer_agent_graph(store, user_id)

    builder = StateGraph(AgentState)
    builder.add_node("report_supervisor", report_supervisor)
    builder.add_node("writer_agent", writer_agent_graph)
    builder.add_node("section_reviewer", section_reviewer)
    builder.add_node("report_assembler", report_assembler)

    builder.add_edge(START, "report_supervisor")
    # report_supervisor uses Send() — no static edge to writer_agent
    builder.add_edge("writer_agent", "section_reviewer")
    # section_reviewer uses Command for routing
    builder.add_edge("report_assembler", END)

    return builder.compile()  # No checkpointer
```

---

## 7.7 Section Content Collection Pattern

When writer agents complete (via `Done` signal), their produced section content must be collected into `state["report_sections"]`. Two patterns:

**Option A (Recommended):** Writer agent returns `{"report_sections": {section_id: {"title": ..., "content": ..., "instructions": ...}}}`. Since `report_sections` is a dict, LangGraph merges it (adds/updates keys, doesn't replace the whole dict — requires custom reducer in state).

**Option B:** Writer agent appends to `messages` with a special marker. The `section_reviewer` parses messages to extract sections.

For Option A, add a custom reducer in `schemas.py`:
```python
from typing import Annotated
from operator import ior

def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

# In AgentState:
report_sections: Annotated[dict, merge_dicts]
```

---

## Completion Checklist

- [ ] `WriterAgentState` TypedDict added to `schemas.py`
- [ ] `report_supervisor` dispatch node written
- [ ] `section_reviewer` node written
- [ ] `writer_agent.py` mini graph built
- [ ] `report_assembler` node written
- [ ] `report_graph.py` wires all nodes
- [ ] Section content collection pattern decided and implemented
- [ ] `report_sections` dict reducer added to `AgentState` if using Option A
- [ ] Report assembler tested: produces readable, well-formatted markdown
