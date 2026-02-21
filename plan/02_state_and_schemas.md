# Plan Step 02 — State & Schemas

## Goal
Define all state classes, Pydantic models, and TypedDict schemas in `schemas.py`. This is the foundation that every other module imports from. Get this right before writing any logic.

---

## 2.1 File: `src/strategic_analyst/schemas.py`

### Imports

```python
from __future__ import annotations
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
```

---

### 2.1.1 Main Agent State

```python
class AgentState(MessagesState):
    """
    Full state for the strategic analyst agent.
    Extends MessagesState which provides built-in `messages` with
    reducer that appends new messages to existing ones.
    """

    # ── Session Identity ──────────────────────────────────────────
    user_id: str
    session_id: str

    # ── Current Request ───────────────────────────────────────────
    query: str                                     # user's current request

    # ── Context Retrieval ─────────────────────────────────────────
    retrieved_context: list[dict]                  # RAG chunks: [{content, metadata, score}]
    memory_context: dict                           # loaded memory: {namespace: content}

    # ── Planning ──────────────────────────────────────────────────
    research_plan: str                             # structured plan text
    plan_approved: bool

    # ── Research ──────────────────────────────────────────────────
    research_tasks: list[dict]                     # [{task_id, question, context}]
    research_findings: dict                        # {task_id: {answer, sources, confidence}}
    supervisor_summary: str                        # supervisor's synthesis of all findings

    # ── Report Writing ────────────────────────────────────────────
    report_sections: dict                          # {section_name: content}
    report_draft: str                              # assembled full draft
    final_report: str                              # approved final version
    report_format: Literal["markdown", "json", "pdf"]

    # ── Routing / Control Flow ────────────────────────────────────
    current_phase: Literal[
        "init", "planning", "research",
        "discoveries", "reporting", "final", "end"
    ]
    hitl_response_type: str                        # last HITL response type for routing
    supervisor_retry_count: int                    # tracks re-research loops
```

> **Note:** `MessagesState` provides `messages: Annotated[list[AnyMessage], add_messages]`.
> Returning `{"messages": [...]}` from a node **appends** to existing messages.

---

### 2.1.2 Input Schema (restricts what caller can pass in)

```python
class AgentInput(TypedDict):
    """What the graph caller passes in to start a session."""
    user_id: str
    query: str
    session_id: Optional[str]      # auto-generated if not provided
```

---

### 2.1.3 Structured Output — Research Plan

```python
class ResearchTask(BaseModel):
    """A single research sub-task within the overall plan."""
    task_id: str = Field(description="Unique identifier, e.g. 'task_1'")
    question: str = Field(description="The specific question this task must answer")
    data_sources: list[Literal["company_db", "web"]] = Field(
        description="Which data sources to use"
    )
    priority: Literal["high", "medium", "low"] = Field(default="medium")
    dependencies: list[str] = Field(
        default_factory=list,
        description="task_ids that must complete before this task"
    )

class ResearchPlan(BaseModel):
    """Structured research plan produced by the planner agent."""
    title: str = Field(description="Short title for this research engagement")
    objective: str = Field(description="What we are trying to discover or answer")
    background: str = Field(description="Context and why this research matters")
    tasks: list[ResearchTask] = Field(description="Ordered list of research tasks")
    expected_deliverable: str = Field(description="What the final report should contain")
    frameworks: list[str] = Field(
        default_factory=list,
        description="Strategic frameworks to apply (SWOT, Porter's 5, etc.)"
    )
```

---

### 2.1.4 Structured Output — Task Agent Findings

```python
class ResearchFinding(BaseModel):
    """What a single task agent returns to the supervisor."""
    task_id: str
    answer: str = Field(description="Direct answer to the research question")
    evidence: list[str] = Field(description="Key facts, data points, or quotes found")
    sources: list[str] = Field(description="URLs or document references")
    confidence: Literal["high", "medium", "low"]
    gaps: str = Field(
        default="",
        description="What could not be found or remains uncertain"
    )
```

---

### 2.1.5 Structured Output — Research Supervisor Review

```python
class SupervisorReview(BaseModel):
    """Supervisor's evaluation of a task agent's finding."""
    task_id: str
    approved: bool
    critique: str = Field(
        default="",
        description="Specific feedback if not approved — what to look for"
    )
    follow_up_question: str = Field(
        default="",
        description="A more targeted question for the re-dispatched agent"
    )
```

class SupervisorDiscoveries(BaseModel):
    """Top-level synthesis from research supervisor to show user at HITL Gate 2."""
    summary: str = Field(description="2-3 paragraph synthesis of all findings")
    key_discoveries: list[str] = Field(description="Bulleted list of most important discoveries")
    open_questions: list[str] = Field(description="Questions that remain unanswered")
    intelligent_follow_ups: list[str] = Field(
        description="2-4 intelligent questions to ask user about the findings"
    )
    recommended_next_steps: list[str] = Field(description="What the supervisor recommends doing next")
```

---

### 2.1.6 Structured Output — Triage Router

```python
class SessionRouter(BaseModel):
    """Routes the user's input to the appropriate workflow phase."""
    reasoning: str = Field(description="Step-by-step reasoning")
    destination: Literal[
        "new_research",         # start full pipeline
        "follow_up_question",   # simple Q&A on existing research
        "report_revision",      # revise existing report
        "memory_query",         # query the user's memory only
        "greeting"              # first message of session
    ]
```

---

### 2.1.7 Structured Output — Memory Update

```python
class MemoryUpdate(BaseModel):
    """LLM-produced targeted memory update."""
    chain_of_thought: str = Field(description="What changed and why")
    updated_content: str = Field(description="The complete updated memory profile")
```

---

### 2.1.8 Structured Output — Report Metadata

```python
class ReportMetadata(BaseModel):
    """Metadata for storing the report in Supabase."""
    title: str
    topic_tags: list[str]
    project_name: str
    executive_summary: str = Field(description="2-3 sentence summary for the database record")
    frameworks_used: list[str]
```
```

---

## 2.2 Summary of What schemas.py Exports

```python
__all__ = [
    "AgentState",
    "AgentInput",
    "ResearchTask",
    "ResearchPlan",
    "ResearchFinding",
    "SupervisorReview",
    "SupervisorDiscoveries",
    "SessionRouter",
    "MemoryUpdate",
    "ReportMetadata",
]
```

---

## Completion Checklist

- [ ] `AgentState` defined with all fields, correct types
- [ ] `AgentInput` TypedDict defined
- [ ] All Pydantic output schemas defined
- [ ] File imports cleanly: `from strategic_analyst.schemas import AgentState`
- [ ] No circular imports
