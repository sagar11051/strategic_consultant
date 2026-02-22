from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


# ── Reducers ──────────────────────────────────────────────────────────────────

def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: shallow-merge two dicts (b wins on key conflicts).
    Used for research_findings and report_sections so parallel Send() workers
    can each write their own key without overwriting each other."""
    return {**a, **b}


# ── Main Agent State ──────────────────────────────────────────────────────────

class AgentState(MessagesState):
    """
    Full state for the strategic analyst agent.
    Extends MessagesState which provides built-in `messages` with
    reducer that appends new messages to existing ones.
    """

    # ── Session Identity ───────────────────────────────────────────────────────
    user_id: str
    session_id: str

    # ── User Identity (populated by context_loader from memory) ───────────────
    # These are cached here so every node can access them without parsing
    # memory_context. Updated whenever memory_writer detects profile changes.
    user_name: str                                  # e.g. "Arjun Mehta"
    user_role: str                                  # e.g. "Senior Strategy Consultant"
    company_name: str                               # e.g. "Nexus Strategy Partners"

    # ── Current Request ────────────────────────────────────────────────────────
    query: str                                      # user's current request

    # ── Context Retrieval ──────────────────────────────────────────────────────
    retrieved_context: list[dict]                   # RAG chunks: [{content, metadata, score}]
    memory_context: dict                            # loaded memory: {namespace: content}

    # ── Planning ───────────────────────────────────────────────────────────────
    research_plan: str                              # structured plan text
    plan_approved: bool

    # ── Research ───────────────────────────────────────────────────────────────
    research_tasks: list[dict]                      # [{task_id, question, context}]
    # Annotated with merge_dicts so parallel Send() task agents each write their
    # own key without overwriting other agents' findings.
    research_findings: Annotated[dict, merge_dicts]  # {task_id: {answer, sources, confidence}}
    supervisor_summary: str                         # supervisor's synthesis of all findings

    # ── Report Writing ─────────────────────────────────────────────────────────
    # Annotated with merge_dicts so parallel Send() writer agents each write
    # their own section_id key without overwriting other sections.
    report_sections: Annotated[dict, merge_dicts]   # {section_id: {title, content, instructions}}
    report_draft: str                               # assembled full draft
    final_report: str                               # approved final version
    report_format: Literal["markdown", "json", "pdf"]

    # ── Routing / Control Flow ─────────────────────────────────────────────────
    current_phase: Literal[
        "init", "planning", "research",
        "discoveries", "reporting", "final", "end"
    ]
    hitl_response_type: str                         # last HITL response type for routing
    supervisor_retry_count: int                     # tracks re-research loops


# ── Input Schema ──────────────────────────────────────────────────────────────

class AgentInput(TypedDict):
    """What the graph caller passes in to start a session."""
    user_id: str
    query: str
    session_id: Optional[str]       # auto-generated if not provided
    # Optional identity hints — context_loader falls back to memory if absent
    user_name: Optional[str]
    user_role: Optional[str]
    company_name: Optional[str]


# ── Sub-agent State (used as Send() payload — not graph state) ─────────────────

class TaskAgentState(TypedDict):
    """State passed to each research task agent via Send()."""
    task_id: str
    question: str
    data_sources: list          # ["company_db", "web"]
    context: str                # relevant context snippet for this task
    retry_count: int
    supervisor_critique: str    # empty on first attempt; supervisor feedback on retry
    user_id: str
    messages: Annotated[list, add_messages]


class WriterAgentState(TypedDict):
    """State passed to each report writer agent via Send()."""
    section_id: str
    section_title: str
    section_instructions: str   # what this section must cover
    research_findings: dict     # full findings for reference
    supervisor_summary: str
    user_preferences: str       # from memory
    user_profile: str           # from memory
    retry_count: int
    supervisor_critique: str
    user_id: str
    messages: Annotated[list, add_messages]


# ── Structured Output — Research Plan ─────────────────────────────────────────

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


# ── Structured Output — Task Agent Findings ───────────────────────────────────

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


# ── Structured Output — Research Supervisor Review ────────────────────────────

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


class SupervisorDiscoveries(BaseModel):
    """Top-level synthesis from research supervisor to show user at HITL Gate 2."""
    summary: str = Field(description="2-3 paragraph synthesis of all findings")
    key_discoveries: list[str] = Field(description="Bulleted list of most important discoveries")
    open_questions: list[str] = Field(description="Questions that remain unanswered")
    intelligent_follow_ups: list[str] = Field(
        description="2-4 intelligent questions to ask user about the findings"
    )
    recommended_next_steps: list[str] = Field(description="What the supervisor recommends doing next")


# ── Structured Output — Triage Router ────────────────────────────────────────

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


# ── Structured Output — Memory Update ────────────────────────────────────────

class MemoryUpdate(BaseModel):
    """LLM-produced targeted memory update."""
    chain_of_thought: str = Field(description="What changed and why")
    updated_content: str = Field(description="The complete updated memory profile")


# ── Structured Output — Report Metadata ──────────────────────────────────────

class ReportMetadata(BaseModel):
    """Metadata for storing the report in Supabase."""
    title: str
    topic_tags: list[str]
    project_name: str
    executive_summary: str = Field(description="2-3 sentence summary for the database record")
    frameworks_used: list[str]


# ── Report Section Plan (used internally by report supervisor) ────────────────

class ReportSection(BaseModel):
    section_id: str
    title: str
    instructions: str = Field(description="What this section must cover, what data to use, tone")
    word_count_target: int = Field(default=400)


class ReportStructure(BaseModel):
    """Planned structure of the report."""
    sections: list[ReportSection]
    formatting_notes: str = Field(
        description="Global formatting instructions based on user preferences"
    )


__all__ = [
    "merge_dicts",
    "AgentState",
    "AgentInput",
    "TaskAgentState",
    "WriterAgentState",
    "ResearchTask",
    "ResearchPlan",
    "ResearchFinding",
    "SupervisorReview",
    "SupervisorDiscoveries",
    "SessionRouter",
    "MemoryUpdate",
    "ReportMetadata",
    "ReportSection",
    "ReportStructure",
]
