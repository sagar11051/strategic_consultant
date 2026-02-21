# Plan Step 10 — Main Graph, langgraph.json & Testing

## Goal
Wire everything together in `main_graph.py`, finalise `langgraph.json`, set up LangSmith tracing, and write the testing strategy.

---

## 10.1 File: `src/strategic_analyst/main_graph.py`

This is the top-level graph that wires all nodes and subgraphs together.

```python
import os
import uuid
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.types import Command

from strategic_analyst.schemas import AgentState, AgentInput
from strategic_analyst.nodes.context_loader import context_loader, greeting_node
from strategic_analyst.nodes.planner import planner_agent
from strategic_analyst.nodes.hitl_gates import hitl_plan_gate, hitl_discovery_gate, hitl_final_gate
from strategic_analyst.nodes.report_saver import save_report_node
from strategic_analyst.subgraphs.research.research_graph import build_research_subgraph
from strategic_analyst.subgraphs.report.report_graph import build_report_subgraph


def build_graph(store: BaseStore = None):
    """
    Factory function to build and compile the main graph.

    Args:
        store: BaseStore instance. If None, creates an InMemoryStore.

    Returns:
        Compiled graph ready for invocation.
    """
    if store is None:
        store = InMemoryStore()

    # Build subgraphs (they need store + user_id, but user_id is per-thread)
    # For subgraph factories that need user_id, we inject it at node-call time
    # using a wrapper that reads user_id from state
    research_subgraph = build_research_subgraph(store, user_id="__placeholder__")
    report_subgraph = build_report_subgraph(store, user_id="__placeholder__")

    # ── Build main graph ───────────────────────────────────────────────────────
    builder = StateGraph(AgentState, input=AgentInput)

    # Add all nodes
    builder.add_node("context_loader", context_loader)
    builder.add_node("greeting_node", greeting_node)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("hitl_plan_gate", hitl_plan_gate)
    builder.add_node("research_subgraph", research_subgraph)    # compiled subgraph
    builder.add_node("hitl_discovery_gate", hitl_discovery_gate)
    builder.add_node("report_subgraph", report_subgraph)        # compiled subgraph
    builder.add_node("hitl_final_gate", hitl_final_gate)
    builder.add_node("save_report_node", save_report_node)

    # ── Static edges ───────────────────────────────────────────────────────────
    builder.add_edge(START, "context_loader")
    builder.add_edge("context_loader", "greeting_node")
    builder.add_edge("greeting_node", "planner_agent")
    # planner_agent → hitl_plan_gate (via Command)

    # ── Post-research routing ──────────────────────────────────────────────────
    builder.add_edge("research_subgraph", "hitl_discovery_gate")
    # hitl_discovery_gate → research_subgraph (loop) or report_subgraph (via Command)

    # ── Post-report routing ────────────────────────────────────────────────────
    builder.add_edge("report_subgraph", "hitl_final_gate")
    # hitl_final_gate → multiple destinations (via Command)

    builder.add_edge("save_report_node", END)

    # ── Compile with checkpointer and store ───────────────────────────────────
    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        store=store
    )


# Module-level compiled graph (what langgraph.json points to)
store = InMemoryStore()
graph = build_graph(store=store)
```

---

## 10.2 Full Routing Map

```
START
  ↓ (static)
context_loader
  ↓ (static)
greeting_node
  ↓ (static)
planner_agent
  ↓ Command(goto="hitl_plan_gate")
hitl_plan_gate
  ├─ accept  → research_subgraph
  ├─ edit    → planner_agent
  ├─ respond → planner_agent
  └─ ignore  → END

research_subgraph
  ↓ (static)
hitl_discovery_gate
  ├─ accept  → report_subgraph
  ├─ respond → research_subgraph  (loop)
  └─ ignore  → END

report_subgraph
  ↓ (static)
hitl_final_gate
  ├─ accept       → save_report_node
  ├─ edit         → report_subgraph (re-assemble)
  ├─ respond:
  │    re-research → research_subgraph
  │    re-plan    → planner_agent
  │    format:    → hitl_final_gate (loop)
  │    Q&A        → research_subgraph (answer inline)
  └─ ignore       → END

save_report_node
  ↓ (static)
END
```

---

## 10.3 `langgraph.json`

```json
{
    "dockerfile_lines": [],
    "graphs": {
        "strategic_analyst": "./src/strategic_analyst/main_graph.py:graph"
    },
    "python_version": "3.12",
    "env": ".env",
    "dependencies": ["."]
}
```

---

## 10.4 LangSmith Tracing Setup

In `main_graph.py` or a startup file:

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Strategic Analyst Agent"
# LANGSMITH_API_KEY is loaded from .env via python-dotenv
```

Or in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Strategic Analyst Agent
```

---

## 10.5 Testing Strategy

### Test Setup

```python
# tests/conftest.py
import pytest
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from strategic_analyst.main_graph import build_graph


@pytest.fixture
def test_graph():
    store = InMemoryStore()
    return build_graph(store=store), store


@pytest.fixture
def thread_config():
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "user_id": "test_user_001",
            "session_id": str(uuid.uuid4()),
        }
    }
```

---

### Test 1: Context Loader

```python
# tests/test_context_loader.py
import pytest
from langgraph.store.memory import InMemoryStore
from strategic_analyst.nodes.context_loader import context_loader
from strategic_analyst.schemas import AgentState

@pytest.mark.asyncio
async def test_context_loader_initialises_memory():
    store = InMemoryStore()
    state = {"user_id": "u1", "query": "market analysis", "messages": []}
    config = {"configurable": {"user_id": "u1", "session_id": "s1"}}

    result = await context_loader(state, config, store)

    assert "memory_context" in result
    assert "user_profile" in result["memory_context"]
    assert "company_profile" in result["memory_context"]
    assert "user_preferences" in result["memory_context"]
    assert "episodic_memory" in result["memory_context"]
    assert result["user_id"] == "u1"
```

---

### Test 2: Memory System

```python
# tests/test_memory.py
import pytest
from langgraph.store.memory import InMemoryStore
from strategic_analyst.memory import (
    load_all_memory, write_memory, get_memory, user_profile_ns
)

def test_memory_initialises_with_defaults():
    store = InMemoryStore()
    memory = load_all_memory(store, "user_001")
    assert "User profile not yet established" in memory["user_profile"]

def test_memory_persists_across_calls():
    store = InMemoryStore()
    ns = user_profile_ns("user_001")
    write_memory(store, ns, "profile", "Name: Test User")
    result = get_memory(store, ns, "profile")
    assert result == "Name: Test User"
```

---

### Test 3: Planner Agent

```python
# tests/test_planner.py
import pytest
from langgraph.store.memory import InMemoryStore
from strategic_analyst.nodes.planner import planner_agent

@pytest.mark.asyncio
async def test_planner_produces_research_plan():
    store = InMemoryStore()
    state = {
        "user_id": "u1",
        "query": "Analyse the APAC fintech market",
        "retrieved_context": [],
        "memory_context": {
            "user_profile": "Name: Test User. Role: Strategy Consultant.",
            "company_profile": "Financial services consulting firm.",
            "user_preferences": "Prefer Porter's Five Forces framework.",
            "episodic_memory": "No prior sessions.",
        },
        "messages": [],
    }
    config = {"configurable": {"user_id": "u1", "model_name": "claude-sonnet-4-6"}}

    result = await planner_agent(state, config, store)

    assert result.goto == "hitl_plan_gate"
    assert "research_plan" in result.update
    assert len(result.update.get("research_tasks", [])) >= 2
```

---

### Test 4: HITL Plan Gate — Accept Path

```python
# tests/test_hitl.py
import pytest
from unittest.mock import patch
from langgraph.store.memory import InMemoryStore

@pytest.mark.asyncio
async def test_hitl_plan_gate_accept():
    store = InMemoryStore()
    state = {
        "user_id": "u1",
        "research_plan": "# Test Plan\n\nTask 1: ...",
        "research_tasks": [{"task_id": "task_1", "question": "Test?", "data_sources": ["web"]}],
        "memory_context": {"user_preferences": ""},
        "messages": [],
    }
    config = {"configurable": {"user_id": "u1", "utility_model_name": "claude-haiku-4-5-20251001"}}

    # Mock interrupt to return "accept"
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept", "args": ""}]
        result = await hitl_plan_gate(state, config, store)

    assert result.goto == "research_subgraph"
    assert result.update["plan_approved"] is True
```

---

### Test 5: Full Graph Smoke Test (with mocked LLM)

```python
# tests/test_integration.py
import pytest
import uuid
from unittest.mock import AsyncMock, patch
from strategic_analyst.main_graph import build_graph
from langgraph.store.memory import InMemoryStore

@pytest.mark.asyncio
async def test_graph_reaches_first_interrupt():
    store = InMemoryStore()
    g, _ = build_graph(store=store), store
    thread_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "user_id": "test_user",
        }
    }

    # Run until first interrupt (hitl_plan_gate)
    result = None
    for chunk in g.stream(
        {"user_id": "test_user", "query": "Market analysis for APAC fintech"},
        config=thread_config
    ):
        result = chunk

    # The graph should be interrupted at hitl_plan_gate
    state = g.get_state(thread_config)
    assert "hitl_plan_gate" in state.next
```

---

## 10.6 Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with LangSmith logging
LANGSMITH_TEST_SUITE="Strategic Analyst" uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_memory.py -v

# Run integration tests only
uv run pytest tests/test_integration.py -v -s
```

---

## 10.7 Manual Testing via Python

```python
import asyncio
import uuid
from strategic_analyst.main_graph import build_graph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

store = InMemoryStore()
graph = build_graph(store=store)

thread_config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "user_id": "arjun_mehta",
    }
}

# Start session
for chunk in graph.stream(
    {"user_id": "arjun_mehta", "query": "I need a market entry analysis for APAC fintech"},
    config=thread_config,
    stream_mode="values"
):
    print(chunk.get("messages", [])[-1] if chunk.get("messages") else "")

# Check what's waiting
state = graph.get_state(thread_config)
print("Waiting at:", state.next)

# Resume with plan approval
for chunk in graph.stream(
    Command(resume=[{"type": "accept", "args": ""}]),
    config=thread_config,
    stream_mode="values"
):
    print(chunk.get("messages", [])[-1] if chunk.get("messages") else "")

# Resume with discovery feedback
for chunk in graph.stream(
    Command(resume=[{"type": "respond", "args": "Go deeper on Singapore's regulatory landscape"}]),
    config=thread_config,
    stream_mode="values"
):
    print(chunk.get("messages", [])[-1] if chunk.get("messages") else "")
```

---

## Completion Checklist

- [ ] `main_graph.py` written and imports all nodes/subgraphs correctly
- [ ] `graph = build_graph()` at module level for LangGraph CLI
- [ ] `langgraph.json` updated to point to `main_graph.py:graph`
- [ ] LangSmith tracing env vars set in `.env`
- [ ] `langgraph dev` starts without errors
- [ ] LangGraph Studio shows graph structure correctly
- [ ] All test files written
- [ ] `uv run pytest tests/` passes all unit tests
- [ ] Manual smoke test: graph reaches first HITL gate and pauses correctly
- [ ] Manual smoke test: resume with `Command(resume=[...])` routes correctly
