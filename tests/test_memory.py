"""
test_memory.py — Unit tests for the memory system.
"""

import pytest
from langgraph.store.memory import InMemoryStore

from strategic_analyst.memory import (
    get_memory,
    load_all_memory,
    user_profile_ns,
    write_memory,
)


def test_memory_initialises_with_defaults():
    store = InMemoryStore()
    memory = load_all_memory(store, "user_001")
    assert isinstance(memory, dict)
    assert "user_profile" in memory
    assert "company_profile" in memory
    assert "user_preferences" in memory
    assert "episodic_memory" in memory
    # Default content is non-empty
    assert memory["user_profile"]
    assert memory["company_profile"]


def test_memory_persists_across_calls():
    store = InMemoryStore()
    ns = user_profile_ns("user_001")
    write_memory(store, ns, "profile", "Name: Test User. Role: Analyst.")
    result = get_memory(store, ns, "profile")
    assert result == "Name: Test User. Role: Analyst."


def test_load_all_memory_uses_written_values():
    store = InMemoryStore()
    ns = user_profile_ns("user_002")
    write_memory(store, ns, "profile", "Name: Jane Doe.")
    memory = load_all_memory(store, "user_002")
    assert "Jane Doe" in memory["user_profile"]


def test_memory_isolated_per_user():
    store = InMemoryStore()
    ns_a = user_profile_ns("user_a")
    ns_b = user_profile_ns("user_b")
    write_memory(store, ns_a, "profile", "User A")
    write_memory(store, ns_b, "profile", "User B")
    assert get_memory(store, ns_a, "profile") == "User A"
    assert get_memory(store, ns_b, "profile") == "User B"
