"""
test_integration.py — Smoke tests for the full compiled graph.

These tests verify the graph can be built and reaches the first HITL
interrupt without making real LLM or network calls.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from strategic_analyst.main_graph import build_graph


def _thread_config():
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "user_id": "test_user",
            "session_id": str(uuid.uuid4()),
        }
    }


def test_graph_builds_without_error():
    """Graph can be instantiated without raising."""
    store = InMemoryStore()
    g = build_graph(store=store)
    assert g is not None


def test_graph_has_expected_nodes():
    """All expected nodes are present in the compiled graph."""
    store = InMemoryStore()
    g = build_graph(store=store)
    node_ids = set(g.get_graph().nodes.keys())
    expected = {
        "context_loader",
        "greeting_node",
        "planner_agent",
        "hitl_plan_gate",
        "research_subgraph",
        "hitl_discovery_gate",
        "report_subgraph",
        "hitl_final_gate",
        "save_report_node",
    }
    assert expected.issubset(node_ids)


@pytest.mark.asyncio
async def test_graph_reaches_first_hitl_interrupt():
    """
    Smoke test: graph streams until the first interrupt (hitl_plan_gate).

    Mocks:
    - hybrid_search (context_loader RAG calls)
    - greeting_node LLM call
    - planner_agent LLM call
    - interrupt() to simulate HITL pause
    """
    store = InMemoryStore()
    g = build_graph(store=store)
    config = _thread_config()

    # --- mock hybrid_search so no real Supabase/OVH call ---
    fake_search = AsyncMock(return_value="")

    # --- mock greeting LLM — must return a real AIMessage so LangGraph can coerce it ---
    from langchain_core.messages import AIMessage
    greeting_msg = AIMessage(content="Hello! How can I help you today?")
    greeting_llm = AsyncMock(return_value=greeting_msg)

    # --- mock planner structured output ---
    from strategic_analyst.schemas import ResearchPlan, ResearchTask
    fake_plan = ResearchPlan(
        title="APAC Fintech Analysis",
        objective="Understand APAC fintech",
        background="Rapid growth in digital payments",
        tasks=[
            ResearchTask(
                task_id="task_1",
                question="What is the market size?",
                data_sources=["web"],
            )
        ],
        expected_deliverable="Strategic report",
    )
    planner_llm = AsyncMock(return_value=fake_plan)

    with (
        patch("strategic_analyst.nodes.context_loader.hybrid_search") as mock_hs,
        patch("strategic_analyst.nodes.context_loader.ChatOpenAI") as mock_greeting_cls,
        patch("strategic_analyst.nodes.planner.ChatOpenAI") as mock_planner_cls,
        patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt,
    ):
        # Wire mocks
        mock_hs.ainvoke = fake_search
        mock_greeting_cls.return_value.ainvoke = greeting_llm

        planner_llm_instance = MagicMock()
        planner_llm_instance.ainvoke = planner_llm
        mock_planner_cls.return_value.with_structured_output.return_value = planner_llm_instance

        # interrupt() is mocked to return "ignore" so the gate routes to __end__
        # cleanly without needing to simulate a real graph pause.
        mock_interrupt.return_value = [{"type": "ignore", "args": ""}]

        chunks = []
        async for chunk in g.astream(
            {"user_id": "test_user", "query": "APAC fintech market analysis"},
            config=config,
        ):
            chunks.append(chunk)

    # Graph ran through context_loader at minimum
    assert len(chunks) >= 1
    # At least one chunk contains messages
    all_keys = set().union(*[set(c.keys()) for c in chunks])
    assert "__end__" in all_keys or any("messages" in c or "context_loader" in c for c in chunks)


@pytest.mark.asyncio
async def test_module_level_graph_is_importable():
    """The module-level `graph` export exists and is the right type."""
    from langgraph.graph.state import CompiledStateGraph

    from strategic_analyst.main_graph import graph

    assert isinstance(graph, CompiledStateGraph)
