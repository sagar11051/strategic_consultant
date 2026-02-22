"""
test_hitl.py — Unit tests for HITL gate nodes.

Uses unittest.mock.patch to mock interrupt() so tests don't actually pause.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from strategic_analyst.nodes.hitl_gates import (
    hitl_discovery_gate,
    hitl_final_gate,
    hitl_plan_gate,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _plan_state():
    return {
        "user_id": "u1",
        "research_plan": "# Test Plan\n\nResearch the APAC fintech market.",
        "research_tasks": [
            {"task_id": "task_1", "question": "Market size?", "data_sources": ["web"]}
        ],
        "memory_context": {"user_preferences": ""},
        "messages": [],
    }


def _discovery_state():
    return {
        "user_id": "u1",
        "supervisor_summary": "Findings summary here.",
        "research_findings": {
            "task_1": {"answer": "Market is large.", "sources": [], "confidence": "high"}
        },
        "memory_context": {"user_preferences": ""},
        "messages": [],
    }


def _final_state():
    return {
        "user_id": "u1",
        "report_draft": "# Strategic Report\n\nContent here.",
        "report_sections": {"section_1": {"title": "Intro", "content": "Intro text."}},
        "report_format": "markdown",
        "memory_context": {"user_preferences": ""},
        "messages": [],
    }


_config = {"configurable": {"user_id": "u1", "utility_model_name": "Mistral-Nemo-Instruct-2407"}}


# ── Gate 1: hitl_plan_gate ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hitl_plan_gate_accept():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept", "args": ""}]
        result = await hitl_plan_gate(_plan_state(), _config, store)
    assert result.goto == "research_subgraph"
    assert result.update["plan_approved"] is True


@pytest.mark.asyncio
async def test_hitl_plan_gate_ignore():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "ignore", "args": ""}]
        result = await hitl_plan_gate(_plan_state(), _config, store)
    assert result.goto == "__end__"


@pytest.mark.asyncio
async def test_hitl_plan_gate_respond():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt, \
         patch("strategic_analyst.nodes.hitl_gates.trigger_memory_update", new_callable=AsyncMock):
        mock_interrupt.return_value = [{"type": "respond", "args": "Focus on regulatory risks"}]
        result = await hitl_plan_gate(_plan_state(), _config, store)
    assert result.goto == "planner_agent"
    assert any("regulatory" in str(m.get("content", "")) for m in result.update["messages"])


# ── Gate 2: hitl_discovery_gate ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hitl_discovery_gate_accept():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept", "args": ""}]
        result = await hitl_discovery_gate(_discovery_state(), _config, store)
    assert result.goto == "report_subgraph"
    assert result.update["current_phase"] == "reporting"


@pytest.mark.asyncio
async def test_hitl_discovery_gate_ignore():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "ignore", "args": ""}]
        result = await hitl_discovery_gate(_discovery_state(), _config, store)
    assert result.goto == "__end__"


@pytest.mark.asyncio
async def test_hitl_discovery_gate_respond():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt, \
         patch("strategic_analyst.nodes.hitl_gates.trigger_memory_update", new_callable=AsyncMock):
        mock_interrupt.return_value = [{"type": "respond", "args": "Go deeper on Singapore"}]
        result = await hitl_discovery_gate(_discovery_state(), _config, store)
    assert result.goto == "research_subgraph"


# ── Gate 3: hitl_final_gate ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hitl_final_gate_accept():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt, \
         patch("strategic_analyst.nodes.hitl_gates.trigger_memory_update", new_callable=AsyncMock):
        mock_interrupt.return_value = [{"type": "accept", "args": ""}]
        result = await hitl_final_gate(_final_state(), _config, store)
    assert result.goto == "save_report_node"
    assert result.update["final_report"] != ""


@pytest.mark.asyncio
async def test_hitl_final_gate_ignore():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "ignore", "args": ""}]
        result = await hitl_final_gate(_final_state(), _config, store)
    assert result.goto == "__end__"


@pytest.mark.asyncio
async def test_hitl_final_gate_respond_re_research():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "respond", "args": "re-research: Singapore regulations"}]
        result = await hitl_final_gate(_final_state(), _config, store)
    assert result.goto == "research_subgraph"


@pytest.mark.asyncio
async def test_hitl_final_gate_respond_re_plan():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "respond", "args": "re-plan: focus on Indonesia only"}]
        result = await hitl_final_gate(_final_state(), _config, store)
    assert result.goto == "planner_agent"


@pytest.mark.asyncio
async def test_hitl_final_gate_respond_format_change():
    store = InMemoryStore()
    with patch("strategic_analyst.nodes.hitl_gates.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "respond", "args": "format: json"}]
        result = await hitl_final_gate(_final_state(), _config, store)
    assert result.goto == "hitl_final_gate"
    assert result.update["report_format"] == "json"
