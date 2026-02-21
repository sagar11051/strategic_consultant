# Plan Step 03 — Memory System

## Goal
Implement the four-namespace memory system using LangGraph's `BaseStore`. This is the personalisation backbone of the entire agent. Every node that needs user context calls into this module.

---

## 3.1 Memory Architecture

```
BaseStore (InMemoryStore in dev, Supabase-backed in prod)
│
├── namespace: ("strategic_analyst", {user_id}, "user_profile")
│   key: "profile"
│   value: str  ← natural language profile, e.g.:
│               "Name: Arjun Mehta. Role: Senior Strategy Consultant.
│                Communication style: Direct, data-driven, prefers bullet points.
│                Current projects: Market entry analysis for APAC region.
│                Expertise: M&A, market sizing, competitive intelligence."
│
├── namespace: ("strategic_analyst", {user_id}, "company_profile")
│   key: "profile"
│   value: str  ← e.g.:
│               "Company: Nexus Strategy Partners. Industry: Management Consulting.
│                Key clients: FMCG, Tech, Financial Services.
│                Competitors: McKinsey, BCG, Bain.
│                Internal tools: Salesforce, internal research DB."
│
├── namespace: ("strategic_analyst", {user_id}, "user_preferences")
│   key: "preferences"
│   value: str  ← e.g.:
│               "Report format: Markdown. Verbosity: High - include full data tables.
│                Preferred frameworks: Porter's Five Forces, SWOT, BCG Matrix.
│                Citation style: Inline with source URL.
│                Meeting notes: Prefers action items bolded."
│
└── namespace: ("strategic_analyst", {user_id}, "episodic_memory")
    key: "episodes"
    value: str  ← chronological research notes, e.g.:
                "2026-02-15: Analysed Q4 earnings for competitor X. Found 23% YoY growth.
                 2026-02-18: Client asked specifically about APAC fintech landscape.
                 2026-02-19: [current session] Market entry research for Region Y."
```

---

## 3.2 File: `src/strategic_analyst/memory.py`

### Default Memory Content (first session initialisation)

```python
DEFAULT_USER_PROFILE = """
User profile not yet established. This is a new user.
Learn their name, role, communication style, and current projects
from the conversation and update this profile accordingly.
"""

DEFAULT_COMPANY_PROFILE = """
Company profile not yet established.
Learn the company name, industry, key priorities, and domain vocabulary
from the conversation and update this profile accordingly.
"""

DEFAULT_USER_PREFERENCES = """
User preferences not yet established.
Default settings:
- Report format: Markdown
- Verbosity: Medium
- Frameworks: SWOT, Porter's Five Forces
- Citation style: Inline source references
Update as the user gives feedback on reports and research.
"""

DEFAULT_EPISODIC_MEMORY = """
No episodic memories yet. This is the first session.
Record key research discoveries, important dates, and temporal context here.
"""
```

---

### Memory Namespace Constants

```python
from typing import Tuple

def user_profile_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "user_profile")

def company_profile_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "company_profile")

def user_preferences_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "user_preferences")

def episodic_memory_ns(user_id: str) -> Tuple[str, str, str]:
    return ("strategic_analyst", user_id, "episodic_memory")

NAMESPACE_KEYS = {
    "user_profile": "profile",
    "company_profile": "profile",
    "user_preferences": "preferences",
    "episodic_memory": "episodes",
}
```

---

### Read Helper

```python
from langgraph.store.base import BaseStore

def get_memory(
    store: BaseStore,
    namespace: tuple,
    key: str,
    default_content: str = ""
) -> str:
    """
    Read a memory value from the store.
    If it doesn't exist, initialise it with default_content.
    Returns the stored string value.
    """
    item = store.get(namespace, key)
    if item:
        return item.value

    # First time: initialise with default
    store.put(namespace, key, default_content)
    return default_content


def load_all_memory(store: BaseStore, user_id: str) -> dict:
    """
    Load all four memory namespaces for a user.
    Returns a dict with namespace type as key and content as value.
    Call this at session start (in context_loader node).
    """
    return {
        "user_profile": get_memory(
            store, user_profile_ns(user_id), "profile", DEFAULT_USER_PROFILE
        ),
        "company_profile": get_memory(
            store, company_profile_ns(user_id), "profile", DEFAULT_COMPANY_PROFILE
        ),
        "user_preferences": get_memory(
            store, user_preferences_ns(user_id), "preferences", DEFAULT_USER_PREFERENCES
        ),
        "episodic_memory": get_memory(
            store, episodic_memory_ns(user_id), "episodes", DEFAULT_EPISODIC_MEMORY
        ),
    }
```

---

### Write Helper (Direct — for simple updates)

```python
def write_memory(
    store: BaseStore,
    namespace: tuple,
    key: str,
    content: str
) -> None:
    """Direct write to store. Use only for simple, non-LLM updates."""
    store.put(namespace, key, content)
```

---

### LLM-Driven Memory Update (for personalisation updates)

```python
from langchain_anthropic import ChatAnthropic
from strategic_analyst.schemas import MemoryUpdate
from strategic_analyst.prompts import MEMORY_UPDATE_SYSTEM_PROMPT
import os

def update_memory_with_llm(
    store: BaseStore,
    namespace: tuple,
    key: str,
    context_messages: list,
    update_reason: str,
    model_name: str = "claude-haiku-4-5-20251001"
) -> None:
    """
    Use an LLM to make targeted updates to a memory namespace.
    Never overwrites — only adds or corrects specific facts.

    Args:
        store: the BaseStore instance
        namespace: memory namespace tuple
        key: the key within the namespace
        context_messages: list of messages providing update context
        update_reason: human-readable reason for why this memory is being updated
        model_name: which Claude model to use (default: Haiku for cost efficiency)
    """
    current_content = get_memory(store, namespace, key)

    llm = ChatAnthropic(
        model=model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ).with_structured_output(MemoryUpdate)

    result: MemoryUpdate = llm.invoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_SYSTEM_PROMPT.format(
                    current_profile=current_content,
                    namespace=str(namespace),
                    update_reason=update_reason,
                )
            }
        ] + context_messages
    )

    store.put(namespace, key, result.updated_content)
```

---

### Async Version (for use in async node functions)

```python
import asyncio
from langchain_anthropic import ChatAnthropic

async def update_memory_with_llm_async(
    store: BaseStore,
    namespace: tuple,
    key: str,
    context_messages: list,
    update_reason: str,
    model_name: str = "claude-haiku-4-5-20251001"
) -> None:
    """Async version of update_memory_with_llm for use in async nodes."""
    current_content = get_memory(store, namespace, key)

    llm = ChatAnthropic(
        model=model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ).with_structured_output(MemoryUpdate)

    result: MemoryUpdate = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_SYSTEM_PROMPT.format(
                    current_profile=current_content,
                    namespace=str(namespace),
                    update_reason=update_reason,
                )
            }
        ] + context_messages
    )

    store.put(namespace, key, result.updated_content)
```

---

## 3.3 Memory Update Triggers (When to Call `update_memory_with_llm`)

| Trigger | Namespace Updated | Update Reason |
|---|---|---|
| HITL Gate 1 — user edits plan | `user_preferences` | "User modified the research plan. Update preferred depth, frameworks, scope." |
| HITL Gate 1 — user rejects plan | `user_preferences` | "User rejected plan. Update preferred research approach." |
| HITL Gate 2 — user asks follow-up | `episodic_memory` | "User showed interest in specific topic. Add temporal note." |
| HITL Gate 2 — user ignores discovery | `user_preferences` | "User indicated this finding is not relevant. Update relevance preferences." |
| HITL Gate 3 — user edits report section | `user_preferences` | "User edited report. Update style and format preferences." |
| HITL Gate 3 — user approves report | `episodic_memory` | "Session completed. Add session summary and key findings as episodic memory." |
| User mentions their name/role | `user_profile` | "User revealed personal context. Update profile." |
| User mentions company info | `company_profile` | "User revealed company context. Update company profile." |
| Research finds new company info | `company_profile` | "Research uncovered new company context." |

---

## 3.4 Memory Search Tool

For the memory search tool (used by supervisor agents to query memory on demand):

```python
def search_memory(
    store: BaseStore,
    user_id: str,
    namespace_type: str,  # "user_profile" | "company_profile" | "user_preferences" | "episodic_memory"
    query: str
) -> str:
    """
    Retrieve a specific memory namespace for the user.
    In future, this can be upgraded to semantic search within the namespace.
    For now, returns the full content of the requested namespace.
    """
    ns_map = {
        "user_profile": (user_profile_ns(user_id), "profile"),
        "company_profile": (company_profile_ns(user_id), "profile"),
        "user_preferences": (user_preferences_ns(user_id), "preferences"),
        "episodic_memory": (episodic_memory_ns(user_id), "episodes"),
    }
    namespace, key = ns_map[namespace_type]
    return get_memory(store, namespace, key)
```

---

## Completion Checklist

- [ ] `memory.py` written with all helpers
- [ ] Default memory strings written in `prompts.py` (MEMORY_UPDATE_SYSTEM_PROMPT)
- [ ] `load_all_memory()` works correctly for a new user (initialises defaults)
- [ ] `load_all_memory()` works correctly for a returning user (returns existing)
- [ ] `update_memory_with_llm()` produces targeted updates (test with a sample message)
- [ ] Namespace functions correctly keyed by `user_id`
