"""
Memory system for the strategic analyst agent.

Four namespaces per user:
  - user_profile      → name, role, communication style, current projects
  - company_profile   → company name, industry, domain vocabulary
  - user_preferences  → report format, verbosity, frameworks, citation style
  - episodic_memory   → chronological research notes and temporal context

All reads initialise the namespace with defaults on first access.
All personalisation writes go through update_memory_with_llm (Haiku model)
to perform targeted updates rather than full overwrites.
"""

from __future__ import annotations

import os
from typing import Tuple

from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI

from strategic_analyst.schemas import MemoryUpdate
from strategic_analyst.prompts import (
    DEFAULT_USER_PROFILE,
    DEFAULT_COMPANY_PROFILE,
    DEFAULT_USER_PREFERENCES,
    DEFAULT_EPISODIC_MEMORY,
    MEMORY_UPDATE_SYSTEM_PROMPT,
)


# ── Namespace Helpers ─────────────────────────────────────────────────────────

def user_profile_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "user_profile")


def company_profile_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "company_profile")


def user_preferences_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "user_preferences")


def episodic_memory_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "episodic_memory")


# Maps namespace type string → (namespace_fn, key, default_content)
_NS_CONFIG = {
    "user_profile": (user_profile_ns, "profile", DEFAULT_USER_PROFILE),
    "company_profile": (company_profile_ns, "profile", DEFAULT_COMPANY_PROFILE),
    "user_preferences": (user_preferences_ns, "preferences", DEFAULT_USER_PREFERENCES),
    "episodic_memory": (episodic_memory_ns, "episodes", DEFAULT_EPISODIC_MEMORY),
}

NAMESPACE_KEYS = {
    "user_profile": "profile",
    "company_profile": "profile",
    "user_preferences": "preferences",
    "episodic_memory": "episodes",
}


# ── Read Helpers ──────────────────────────────────────────────────────────────

def get_memory(
    store: BaseStore,
    namespace: tuple,
    key: str,
    default_content: str = "",
) -> str:
    """
    Read a memory value from the store.
    If not present, initialise with default_content and return it.
    """
    item = store.get(namespace, key)
    if item is not None:
        # BaseStore items expose value via .value
        return item.value if hasattr(item, "value") else str(item)

    # First access — initialise with default
    store.put(namespace, key, default_content)
    return default_content


def load_all_memory(store: BaseStore, user_id: str) -> dict:
    """
    Load all four memory namespaces for a user in one call.
    Returns {namespace_type: content_string}.
    Call this at session start inside the context_loader node.
    """
    result: dict = {}
    for ns_type, (ns_fn, key, default) in _NS_CONFIG.items():
        result[ns_type] = get_memory(store, ns_fn(user_id), key, default)
    return result


def search_memory(
    store: BaseStore,
    user_id: str,
    namespace_type: str,
    query: str,  # reserved for future semantic search; currently unused
) -> str:
    """
    Retrieve a specific memory namespace for the user.
    Currently returns the full content; can be upgraded to semantic
    search within the namespace in future.
    """
    if namespace_type not in _NS_CONFIG:
        raise ValueError(
            f"Unknown namespace_type '{namespace_type}'. "
            f"Choose from: {list(_NS_CONFIG)}"
        )
    ns_fn, key, default = _NS_CONFIG[namespace_type]
    return get_memory(store, ns_fn(user_id), key, default)


# ── Write Helpers ─────────────────────────────────────────────────────────────

def write_memory(
    store: BaseStore,
    namespace: tuple,
    key: str,
    content: str,
) -> None:
    """
    Direct write to store. Use only for simple, non-LLM updates
    (e.g. clearing a namespace or bootstrapping defaults manually).
    For personalisation updates always use update_memory_with_llm.
    """
    store.put(namespace, key, content)


# ── LLM-Driven Memory Update ──────────────────────────────────────────────────

def update_memory_with_llm(
    store: BaseStore,
    namespace: tuple,
    key: str,
    context_messages: list,
    update_reason: str,
    model_name: str = "Mistral-Nemo-Instruct-2407",
) -> None:
    """
    Use an OVH-hosted LLM to make a targeted update to a memory namespace.
    Never overwrites blindly — the LLM integrates new info while
    preserving all existing accurate content.

    Args:
        store:            the BaseStore instance
        namespace:        memory namespace tuple
        key:              the key within the namespace
        context_messages: list of LangChain message dicts providing context
        update_reason:    human-readable reason this memory is being updated
        model_name:       model to use (default: Mistral-Nemo via OVH)
    """
    current_content = get_memory(store, namespace, key)

    llm = ChatOpenAI(
        model=os.getenv("UTILITY_MODEL", model_name),
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
        temperature=0.0,
    ).with_structured_output(MemoryUpdate)

    result: MemoryUpdate = llm.invoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_SYSTEM_PROMPT.format(
                    current_profile=current_content,
                    namespace=str(namespace),
                    update_reason=update_reason,
                ),
            }
        ]
        + context_messages
    )

    store.put(namespace, key, result.updated_content)


async def update_memory_with_llm_async(
    store: BaseStore,
    namespace: tuple,
    key: str,
    context_messages: list,
    update_reason: str,
    model_name: str = "Mistral-Nemo-Instruct-2407",
) -> None:
    """
    Async version of update_memory_with_llm for use inside async node functions.
    Signature is identical — use this inside any `async def` node.
    """
    current_content = get_memory(store, namespace, key)

    llm = ChatOpenAI(
        model=os.getenv("UTILITY_MODEL", model_name),
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
        temperature=0.0,
    ).with_structured_output(MemoryUpdate)

    result: MemoryUpdate = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_SYSTEM_PROMPT.format(
                    current_profile=current_content,
                    namespace=str(namespace),
                    update_reason=update_reason,
                ),
            }
        ]
        + context_messages
    )

    store.put(namespace, key, result.updated_content)
