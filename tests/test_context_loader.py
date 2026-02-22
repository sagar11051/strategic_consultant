"""
test_context_loader.py — Unit tests for the context loader node.

These tests mock the OVH embedding endpoint so no real network call is made.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from strategic_analyst.nodes.context_loader import context_loader


@pytest.mark.asyncio
async def test_context_loader_initialises_state():
    store = InMemoryStore()
    state = {"user_id": "u1", "query": "market analysis", "messages": []}
    config = {"configurable": {"user_id": "u1", "session_id": "s1"}}

    # Patch hybrid_search so no real Supabase/OVH call is made
    with patch(
        "strategic_analyst.nodes.context_loader.hybrid_search",
        new_callable=AsyncMock,
    ) as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="")
        result = await context_loader(state, config, store)

    assert "memory_context" in result
    assert "user_profile" in result["memory_context"]
    assert "company_profile" in result["memory_context"]
    assert "user_preferences" in result["memory_context"]
    assert "episodic_memory" in result["memory_context"]
    assert result["user_id"] == "u1"
    assert result["current_phase"] == "init"
    assert result["plan_approved"] is False


@pytest.mark.asyncio
async def test_context_loader_seeds_identity_from_state():
    store = InMemoryStore()
    state = {
        "user_id": "u2",
        "query": "test query",
        "user_name": "Alice",
        "user_role": "Consultant",
        "company_name": "Acme Corp",
        "messages": [],
    }
    config = {"configurable": {"user_id": "u2", "session_id": "s2"}}

    with patch(
        "strategic_analyst.nodes.context_loader.hybrid_search",
        new_callable=AsyncMock,
    ) as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="")
        result = await context_loader(state, config, store)

    assert result["user_name"] == "Alice"
    assert result["user_role"] == "Consultant"
    assert result["company_name"] == "Acme Corp"
