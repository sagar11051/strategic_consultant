"""
conftest.py — Shared pytest fixtures for the strategic analyst agent tests.
"""

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from strategic_analyst.main_graph import build_graph


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def test_graph(store):
    """Returns a compiled graph using MemorySaver + InMemoryStore."""
    return build_graph(store=store, checkpointer=MemorySaver())


@pytest.fixture
def thread_config():
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "user_id": "test_user_001",
            "session_id": str(uuid.uuid4()),
        }
    }
