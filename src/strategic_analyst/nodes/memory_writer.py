"""
memory_writer.py — Memory update utility for graph nodes.

Not a graph node itself. Called from within other nodes (e.g. report_saver,
hitl_gates) to persist learnings to the user's memory namespaces.
"""

from __future__ import annotations

from langgraph.store.base import BaseStore

from strategic_analyst.memory import (
    NAMESPACE_KEYS,
    company_profile_ns,
    episodic_memory_ns,
    update_memory_with_llm_async,
    user_preferences_ns,
    user_profile_ns,
)


async def trigger_memory_update(
    store: BaseStore,
    user_id: str,
    namespace_type: str,
    context_messages: list,
    update_reason: str,
    model_name: str = "Mistral-Nemo-Instruct-2407",
) -> None:
    """
    Convenience wrapper that triggers an LLM-driven memory update.

    Args:
        store:             the BaseStore instance
        user_id:           identifies the user's memory partition
        namespace_type:    one of "user_profile", "company_profile",
                           "user_preferences", "episodic_memory"
        context_messages:  list of message dicts providing new context
        update_reason:     human-readable reason for this update
        model_name:        OVH-hosted model to use (default: Mistral-Nemo)
    """
    _ns_map = {
        "user_profile": (user_profile_ns(user_id), NAMESPACE_KEYS["user_profile"]),
        "company_profile": (company_profile_ns(user_id), NAMESPACE_KEYS["company_profile"]),
        "user_preferences": (user_preferences_ns(user_id), NAMESPACE_KEYS["user_preferences"]),
        "episodic_memory": (episodic_memory_ns(user_id), NAMESPACE_KEYS["episodic_memory"]),
    }

    if namespace_type not in _ns_map:
        raise ValueError(
            f"Unknown namespace_type '{namespace_type}'. "
            f"Choose from: {list(_ns_map)}"
        )

    namespace, key = _ns_map[namespace_type]
    await update_memory_with_llm_async(
        store=store,
        namespace=namespace,
        key=key,
        context_messages=context_messages,
        update_reason=update_reason,
        model_name=model_name,
    )
