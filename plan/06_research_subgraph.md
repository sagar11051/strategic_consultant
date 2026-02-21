# Plan Step 06 — Research Subgraph

## Goal
Build the autonomous multi-agent research pipeline. This is the most complex subgraph. It uses LangGraph's `Send()` API for parallel task dispatch, a supervisor review loop, and a synthesis step before returning to the main graph.

---

## 6.1 Subgraph Structure

```
research_graph (StateGraph)
       │
       ▼
[research_supervisor]
   ├── Reads: research_plan, research_tasks, retrieved_context, memory_context
   ├── Decomposes tasks
   └── Dispatches via Send() → parallel fan-out
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
[task_agent_1] [task_agent_2] [task_agent_N]
    │        │        │
    └────────┴────────┘
             │ all findings return via state
             ▼
[supervisor_review]
   ├── Evaluates each finding
   ├── Approves → continue
   └── Rejects → re-dispatch via Send()
             │
             ▼
[discovery_synthesiser]
   ├── Builds supervisor_summary
   ├── Writes episodic_memory
   └── → END (returns to main graph)
```

---

## 6.2 State for Research Subgraph

The research subgraph uses the same `AgentState`. However, task agents get their own mini-state for the `Send()` payload:

```python
# In schemas.py — add this
class TaskAgentState(TypedDict):
    """State passed to each task agent via Send()."""
    task_id: str
    question: str
    data_sources: list[str]
    context: str             # relevant chunks for this task
    retry_count: int
    supervisor_critique: str  # empty on first attempt, filled on retry
    # Plus the full agent state fields needed by tools:
    user_id: str
    messages: list           # starts empty for each task agent
```

---

## 6.3 File: `src/strategic_analyst/subgraphs/research/supervisor.py`

### Research Supervisor — Initial Dispatch Node

```python
import os
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send

from strategic_analyst.schemas import AgentState, SupervisorDiscoveries
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import RESEARCH_SUPERVISOR_SYSTEM_PROMPT
from strategic_analyst.tools.base import get_tools, RESEARCH_SUPERVISOR_TOOLS


async def research_supervisor(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command:
    """
    Research supervisor node — entry point of the research subgraph.

    Actions:
    1. Load memory context (already in state from context_loader)
    2. Analyse the research plan and task list
    3. Dispatch all tasks in parallel via Send()
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    tasks = state.get("research_tasks", [])
    context_chunks = state.get("retrieved_context", [])
    memory = state.get("memory_context", {})

    # Assign relevant context to each task
    task_sends = []
    for task in tasks:
        # Give each task agent its own relevant context slice
        # In a more sophisticated version, filter chunks by task relevance
        task_context = "\n\n".join(
            chunk.get("content", "") for chunk in context_chunks[:3]
        )

        task_sends.append(
            Send("task_agent", {
                "task_id": task["task_id"],
                "question": task["question"],
                "data_sources": task.get("data_sources", ["company_db", "web"]),
                "context": task_context,
                "retry_count": 0,
                "supervisor_critique": "",
                "user_id": state["user_id"],
                "messages": [],
            })
        )

    return Command(
        goto=task_sends,  # Fan-out: dispatch all tasks in parallel
        update={
            "current_phase": "research",
            "messages": [{
                "role": "assistant",
                "content": f"Starting research with {len(tasks)} parallel tasks..."
            }]
        }
    )
```

### Research Supervisor — Review Node (after task agents return)

```python
from strategic_analyst.schemas import SupervisorReview, ResearchFinding
import json


async def supervisor_review(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command[Literal["task_agent", "discovery_synthesiser"]]:
    """
    Evaluates all task agent findings.
    - Approved findings → accumulate in research_findings
    - Rejected findings → re-dispatch that task via Send()

    This node runs AFTER all task agents have returned.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    findings = state.get("research_findings", {})
    tasks = state.get("research_tasks", [])
    retry_count = state.get("supervisor_retry_count", 0)

    # Check if we've hit max retries
    if retry_count >= cfg.max_supervisor_retries:
        # Accept whatever we have and move on
        return Command(goto="discovery_synthesiser", update={})

    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0
    ).with_structured_output(SupervisorReview)

    re_dispatch = []
    approved_findings = dict(findings)

    for task in tasks:
        task_id = task["task_id"]
        if task_id not in findings:
            # Task didn't return a finding — re-dispatch
            re_dispatch.append(
                Send("task_agent", {
                    "task_id": task_id,
                    "question": task["question"],
                    "data_sources": task.get("data_sources", ["company_db", "web"]),
                    "context": "",
                    "retry_count": 1,
                    "supervisor_critique": "Task did not return any findings. Please retry.",
                    "user_id": state["user_id"],
                    "messages": [],
                })
            )
            continue

        finding = findings[task_id]
        review: SupervisorReview = await llm.ainvoke([
            {"role": "system", "content": SUPERVISOR_REVIEW_PROMPT},
            {"role": "user", "content": (
                f"Task: {task['question']}\n\n"
                f"Finding:\n{json.dumps(finding, indent=2)}"
            )}
        ])

        if not review.approved:
            # Re-dispatch with critique
            re_dispatch.append(
                Send("task_agent", {
                    "task_id": task_id,
                    "question": review.follow_up_question or task["question"],
                    "data_sources": task.get("data_sources", ["company_db", "web"]),
                    "context": "",
                    "retry_count": retry_count + 1,
                    "supervisor_critique": review.critique,
                    "user_id": state["user_id"],
                    "messages": [],
                })
            )

    if re_dispatch:
        return Command(
            goto=re_dispatch,
            update={"supervisor_retry_count": retry_count + 1}
        )
    else:
        return Command(goto="discovery_synthesiser", update={})
```

---

## 6.4 File: `src/strategic_analyst/subgraphs/research/task_agent.py`

```python
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

from strategic_analyst.schemas import TaskAgentState, ResearchFinding
from strategic_analyst.tools.base import get_tools, make_tool_node, TASK_AGENT_TOOLS
from strategic_analyst.prompts import TASK_AGENT_SYSTEM_PROMPT
from strategic_analyst.configuration import AgentConfiguration


def build_task_agent_graph(store: BaseStore, user_id: str):
    """
    Factory that builds a task agent subgraph.
    Each task agent is a mini ReAct loop: LLM call → tool call → LLM call → ... → Done

    The graph:
      START → [task_llm_call] → (tool_calls?) → [task_tool_node] → [task_llm_call]
                                              ↘ (Done) → END
    """
    tools = get_tools(TASK_AGENT_TOOLS)
    tools_by_name = {t.name: t for t in tools}
    llm = ChatAnthropic(
        model=os.getenv("TASK_AGENT_MODEL", "claude-sonnet-4-6"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    ).bind_tools(tools, tool_choice="required")

    async def task_llm_call(state: TaskAgentState, config: RunnableConfig):
        """LLM call for a research task agent."""
        critique = state.get("supervisor_critique", "")
        retry_info = ""
        if critique:
            retry_info = f"\n\n**Supervisor Critique (you must address this):** {critique}"

        system = TASK_AGENT_SYSTEM_PROMPT.format(
            task_id=state["task_id"],
            question=state["question"],
            data_sources=", ".join(state.get("data_sources", [])),
            context=state.get("context", "No prior context"),
            retry_info=retry_info,
        )

        response = await llm.ainvoke(
            [{"role": "system", "content": system}] + state.get("messages", [])
        )
        return {"messages": [response]}

    async def task_tool_node(state: TaskAgentState, config: RunnableConfig):
        """Execute tool calls for the task agent."""
        tool_node_fn = make_tool_node(tools, store, state["user_id"])
        return await tool_node_fn(state)

    def should_continue_task(state: TaskAgentState) -> Literal["task_tool_node", "__end__"]:
        """Route: more tool calls → tool node; Done tool → END."""
        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                if tc["name"] == "Done":
                    return END
            return "task_tool_node"
        return END

    # Build the mini graph
    builder = StateGraph(TaskAgentState)
    builder.add_node("task_llm_call", task_llm_call)
    builder.add_node("task_tool_node", task_tool_node)
    builder.add_edge(START, "task_llm_call")
    builder.add_conditional_edges("task_llm_call", should_continue_task, {
        "task_tool_node": "task_tool_node",
        END: END,
    })
    builder.add_edge("task_tool_node", "task_llm_call")

    return builder.compile()  # No checkpointer — inherits from parent
```

---

## 6.5 Discovery Synthesiser Node

```python
async def discovery_synthesiser(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """
    Synthesises all research findings into a structured discovery summary.
    Also writes episodic memory with key findings.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    findings = state.get("research_findings", {})
    memory = state.get("memory_context", {})

    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.2
    ).with_structured_output(SupervisorDiscoveries)

    findings_text = "\n\n".join(
        f"**{task_id}:** {data.get('answer', '')}"
        for task_id, data in findings.items()
    )

    discoveries: SupervisorDiscoveries = await llm.ainvoke([
        {"role": "system", "content": DISCOVERY_SYNTHESIS_PROMPT.format(
            user_profile=memory.get("user_profile", ""),
            user_preferences=memory.get("user_preferences", ""),
        )},
        {"role": "user", "content": f"Research findings:\n\n{findings_text}"}
    ])

    # Write episodic memory with key findings
    from strategic_analyst.nodes.memory_writer import trigger_memory_update
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await trigger_memory_update(
        store=store,
        user_id=state["user_id"],
        namespace_type="episodic_memory",
        context_messages=[{
            "role": "user",
            "content": f"On {today}, key research discoveries:\n" + "\n".join(discoveries.key_discoveries)
        }],
        update_reason="Research phase completed. Add key discoveries as temporal notes.",
        utility_model=cfg.utility_model_name,
    )

    return {
        "supervisor_summary": discoveries.summary,
        "messages": [{
            "role": "assistant",
            "content": f"**Research Complete**\n\n{discoveries.summary}"
        }]
    }
```

---

## 6.6 File: `src/strategic_analyst/subgraphs/research/research_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from strategic_analyst.schemas import AgentState

from .supervisor import research_supervisor, supervisor_review
from .task_agent import build_task_agent_graph
from .discovery_synthesiser import discovery_synthesiser


def build_research_subgraph(store, user_id: str):
    """
    Factory: build and compile the research subgraph.
    Must be called with store + user_id to inject into task agents.
    """
    task_agent_graph = build_task_agent_graph(store, user_id)

    builder = StateGraph(AgentState)
    builder.add_node("research_supervisor", research_supervisor)
    builder.add_node("task_agent", task_agent_graph)  # subgraph as node
    builder.add_node("supervisor_review", supervisor_review)
    builder.add_node("discovery_synthesiser", discovery_synthesiser)

    builder.add_edge(START, "research_supervisor")
    # research_supervisor uses Send() — no static edge to task_agent
    # After all task_agents complete, they flow to supervisor_review
    builder.add_edge("task_agent", "supervisor_review")
    # supervisor_review uses Command to route to task_agent (re-dispatch) or discovery_synthesiser
    builder.add_edge("discovery_synthesiser", END)

    return builder.compile()  # No checkpointer — inherits from parent
```

---

## 6.7 How Task Agent Results Flow Back

When `Send("task_agent", task_state)` is used, each task agent runs independently. Their results (messages containing the `Done` tool call with the finding) need to be aggregated back into `research_findings` in the main state.

**Implementation:** The task agent's `Done` tool call args contain the structured finding. The `supervisor_review` node reads from `state["messages"]` to find tool calls named `Done` and extracts the `task_id` and finding data. Alternatively, the task agent writes directly to `research_findings[task_id]` via state update.

The simplest approach: task agent's last message contains the `ResearchFinding` as structured output, and the `supervisor_review` node parses all messages to collect findings per `task_id`.

---

## Completion Checklist

- [ ] `TaskAgentState` TypedDict added to `schemas.py`
- [ ] `research_supervisor` dispatch node written with `Send()` fan-out
- [ ] `supervisor_review` node written with approve/reject logic
- [ ] `task_agent.py` mini ReAct graph built correctly
- [ ] `discovery_synthesiser` node written with episodic memory write
- [ ] `research_graph.py` wires all nodes correctly
- [ ] Task agent results correctly flow back to `research_findings` state
- [ ] `Send()` fan-out tested with 2+ tasks
- [ ] Retry logic tested (reject → re-dispatch → approve)
