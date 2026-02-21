# Plan Step 05 — Main Graph Nodes

## Goal
Implement all nodes that live directly in the main graph: context loader, greeting, planner, and report saver. These are the "outer shell" nodes that wrap the two subgraphs.

---

## 5.1 File: `src/strategic_analyst/nodes/context_loader.py`

The very first node. Fires parallel async calls to load memory and RAG context before the agent ever responds to the user.

```python
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.memory import load_all_memory
from strategic_analyst.tools.rag_tool import rag_tool
import uuid


async def context_loader(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """
    Session initialisation node. Runs before anything else.

    Actions:
    1. Load all four memory namespaces for the user (parallel)
    2. Fire initial RAG queries based on the user's query (parallel)
    3. Return combined context to state

    This node does NOT call an LLM — it's purely data loading.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    user_id = cfg.user_id
    session_id = cfg.session_id or str(uuid.uuid4())
    query = state.get("query", "")

    # Generate initial RAG queries from the user's query
    # Use the query directly + a few broadening variants
    rag_queries = [
        query,
        f"strategic context: {query}",
        f"company data related to: {query}",
    ]

    # Parallel execution: memory load + multiple RAG calls
    memory_task = asyncio.create_task(
        asyncio.to_thread(load_all_memory, store, user_id)
    )

    rag_tasks = [
        asyncio.create_task(
            rag_tool.ainvoke({"query": q, "top_k": cfg.rag_top_k // len(rag_queries)})
        )
        for q in rag_queries
    ]

    # Await all
    memory_context = await memory_task
    rag_results = await asyncio.gather(*rag_tasks, return_exceptions=True)

    # Combine RAG results
    retrieved_chunks = []
    for result in rag_results:
        if isinstance(result, Exception):
            continue  # Gracefully skip failed retrievals
        if isinstance(result, str) and result:
            retrieved_chunks.append({"content": result, "source": "company_db"})

    return {
        "user_id": user_id,
        "session_id": session_id,
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
        "report_format": "markdown",
    }
```

---

## 5.2 Greeting Node

```python
# in src/strategic_analyst/nodes/context_loader.py (or a separate greeting.py)

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import GREETING_SYSTEM_PROMPT


async def greeting_node(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """
    Generates a personalised greeting using the loaded memory context.
    Acknowledges the user by name, references their current project if known,
    and asks a focused opening question to understand today's goal.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})

    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3
    )

    system_prompt = GREETING_SYSTEM_PROMPT.format(
        user_profile=memory.get("user_profile", ""),
        company_profile=memory.get("company_profile", ""),
        user_preferences=memory.get("user_preferences", ""),
        episodic_memory=memory.get("episodic_memory", ""),
    )

    greeting = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.get("query", "Hello")}
    ])

    return {
        "messages": [greeting],
        "current_phase": "planning",
    }
```

---

## 5.3 File: `src/strategic_analyst/nodes/planner.py`

```python
import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Command
from typing import Literal

from strategic_analyst.schemas import AgentState, ResearchPlan
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import PLANNER_SYSTEM_PROMPT


async def planner_agent(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> Command[Literal["hitl_plan_gate"]]:
    """
    Research planning node.

    Input:  user query + retrieved_context + memory_context
    Output: structured ResearchPlan → stored as research_plan in state

    Always routes to hitl_plan_gate after producing a plan.
    Re-planning (after user feedback) also goes through this node.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    memory = state.get("memory_context", {})
    context = state.get("retrieved_context", [])
    query = state.get("query", "")

    # Condense retrieved context for the planner
    context_str = "\n\n".join(
        chunk.get("content", "") for chunk in context[:5]  # top 5 chunks
    )

    llm = ChatAnthropic(
        model=cfg.model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1
    ).with_structured_output(ResearchPlan)

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        user_profile=memory.get("user_profile", ""),
        company_profile=memory.get("company_profile", ""),
        user_preferences=memory.get("user_preferences", ""),
        episodic_memory=memory.get("episodic_memory", ""),
        retrieved_context=context_str,
    )

    # Include conversation history so re-planning incorporates feedback
    messages = [{"role": "system", "content": system_prompt}] + state.get("messages", [])
    if not any(m.get("role") == "user" if isinstance(m, dict) else m.type == "human"
               for m in messages[-5:]):
        messages.append({"role": "user", "content": f"Please create a research plan for: {query}"})

    plan: ResearchPlan = await llm.ainvoke(messages)

    # Format plan as readable markdown for display in HITL gate
    plan_markdown = _format_plan_as_markdown(plan)

    return Command(
        goto="hitl_plan_gate",
        update={
            "research_plan": plan_markdown,
            "research_tasks": [task.model_dump() for task in plan.tasks],
            "current_phase": "planning",
            "messages": [{
                "role": "assistant",
                "content": f"I've prepared a research plan for your request:\n\n{plan_markdown}"
            }]
        }
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
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
        lines.append(f"{priority_icon} **{task.task_id}** [{task.priority.upper()}]")
        lines.append(f"   Question: {task.question}")
        lines.append(f"   Sources: {', '.join(task.data_sources)}")
        if task.dependencies:
            lines.append(f"   Depends on: {', '.join(task.dependencies)}")
        lines.append("")

    return "\n".join(lines)
```

---

## 5.4 File: `src/strategic_analyst/nodes/report_saver.py`

```python
import os
import uuid
from datetime import datetime, timezone
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from supabase import create_client
from langchain_anthropic import ChatAnthropic

from strategic_analyst.schemas import AgentState, ReportMetadata
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.memory import (
    update_memory_with_llm_async, episodic_memory_ns, NAMESPACE_KEYS
)
from strategic_analyst.prompts import REPORT_METADATA_SYSTEM_PROMPT


async def save_report_node(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """
    Final node — saves the approved report to Supabase.

    Actions:
    1. Extract metadata from the report using LLM (Haiku)
    2. Store report in `reports` table
    3. Chunk + embed the report and insert into `report_chunks` table
       so it's retrievable in future RAG calls
    4. Update episodic_memory with session summary
    5. Return confirmation message
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    user_id = cfg.user_id
    final_report = state.get("final_report", state.get("report_draft", ""))
    report_format = state.get("report_format", "markdown")
    session_id = state.get("session_id", str(uuid.uuid4()))

    if not final_report:
        return {
            "messages": [{"role": "assistant", "content": "No report content to save."}],
            "current_phase": "end"
        }

    # Step 1: Extract metadata using Claude Haiku
    llm = ChatAnthropic(
        model=cfg.utility_model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ).with_structured_output(ReportMetadata)

    metadata: ReportMetadata = await llm.ainvoke([
        {"role": "system", "content": REPORT_METADATA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract metadata from this report:\n\n{final_report[:3000]}"}
    ])

    # Step 2: Store in Supabase `reports` table
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    report_id = str(uuid.uuid4())
    report_record = {
        "id": report_id,
        "user_id": user_id,
        "session_id": session_id,
        "title": metadata.title,
        "content": final_report,
        "format": report_format,
        "topic_tags": metadata.topic_tags,
        "project_name": metadata.project_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    supabase.table("reports").insert(report_record).execute()

    # Step 3: Chunk report and embed for future RAG
    # Split into ~500 word chunks
    chunks = _chunk_text(final_report, chunk_size=500)
    for i, chunk in enumerate(chunks):
        # NOTE: embed chunk using same embedding model as documents table
        # Placeholder — implement with actual embedding model
        chunk_embedding = await _embed_text(chunk)
        supabase.table("report_chunks").insert({
            "id": str(uuid.uuid4()),
            "report_id": report_id,
            "content": chunk,
            "embedding": chunk_embedding,
            "metadata": {
                "title": metadata.title,
                "user_id": user_id,
                "chunk_index": i,
                "document_type": "generated_report",
                "topic_tags": metadata.topic_tags,
            }
        }).execute()

    # Step 4: Update episodic memory with session summary
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await update_memory_with_llm_async(
        store=store,
        namespace=episodic_memory_ns(user_id),
        key=NAMESPACE_KEYS["episodic_memory"],
        context_messages=[{
            "role": "user",
            "content": (
                f"On {today}, completed a research session on: {metadata.title}. "
                f"Key findings: {metadata.executive_summary}. "
                f"Report saved with ID {report_id}."
            )
        }],
        update_reason=f"Session completed on {today}. Update episodic memory with session summary.",
        model_name=cfg.utility_model_name
    )

    return {
        "messages": [{
            "role": "assistant",
            "content": (
                f"✅ Report saved successfully!\n\n"
                f"**Title:** {metadata.title}\n"
                f"**Report ID:** `{report_id}`\n"
                f"**Format:** {report_format}\n"
                f"**Tags:** {', '.join(metadata.topic_tags)}\n\n"
                f"The report has been added to the company knowledge base "
                f"and will be retrievable in future research sessions."
            )
        }],
        "current_phase": "end",
    }


def _chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


async def _embed_text(text: str) -> list[float]:
    """Embed text using the configured embedding model. Placeholder."""
    # TODO: implement with actual embedding model
    # (must match the model used to index documents table)
    raise NotImplementedError("Implement with actual embedding model after Supabase schema is provided")
```

---

## 5.5 File: `src/strategic_analyst/nodes/memory_writer.py`

Utility node callable from any point in the graph to trigger memory updates.

```python
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Command

from strategic_analyst.schemas import AgentState
from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.memory import (
    update_memory_with_llm_async,
    user_profile_ns, company_profile_ns, user_preferences_ns, episodic_memory_ns,
    NAMESPACE_KEYS
)


async def trigger_memory_update(
    store: BaseStore,
    user_id: str,
    namespace_type: str,
    context_messages: list,
    update_reason: str,
    utility_model: str = "claude-haiku-4-5-20251001"
) -> None:
    """
    Convenience function to trigger a memory update from any node.
    Not a graph node itself — called inside other nodes.
    """
    ns_map = {
        "user_profile": (user_profile_ns(user_id), NAMESPACE_KEYS["user_profile"]),
        "company_profile": (company_profile_ns(user_id), NAMESPACE_KEYS["company_profile"]),
        "user_preferences": (user_preferences_ns(user_id), NAMESPACE_KEYS["user_preferences"]),
        "episodic_memory": (episodic_memory_ns(user_id), NAMESPACE_KEYS["episodic_memory"]),
    }
    namespace, key = ns_map[namespace_type]
    await update_memory_with_llm_async(
        store=store,
        namespace=namespace,
        key=key,
        context_messages=context_messages,
        update_reason=update_reason,
        model_name=utility_model,
    )
```

---

## Completion Checklist

- [ ] `context_loader.py` written with parallel async pattern
- [ ] `greeting_node` written and uses GREETING_SYSTEM_PROMPT
- [ ] `planner.py` written with `ResearchPlan` structured output
- [ ] `planner.py` handles re-planning (reads full message history for feedback)
- [ ] `report_saver.py` written (embedding call stubbed, to be completed)
- [ ] `memory_writer.py` utility function written
- [ ] All nodes have correct type signatures with `store: BaseStore`
