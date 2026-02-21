# PRD — Strategic Analyst Ambient RAG Agent

**Version:** 1.0
**Date:** 2026-02-19
**Status:** Planning

---

## 1. Executive Summary

A production-grade, personalised ambient agent built on LangGraph that acts as a senior strategic analyst for management consultants. The agent:

- Knows the user personally (profile, style, preferences, history)
- Retrieves from the company's Supabase vector database
- Plans research in collaboration with the user
- Dispatches a multi-agent research team with supervisor oversight
- Writes structured strategic reports in the user's preferred style
- Learns continuously from every human-in-the-loop interaction
- Persists all conversation history and memory across sessions

---

## 2. Users & Context

**Primary User:** A management or strategy consultant working at a firm.

**Context:** The consultant uses this agent daily to:
- Conduct market analysis and competitive intelligence
- Analyse internal company documents and research
- Perform gap analysis between current state and strategic targets
- Draft polished research reports for internal/client delivery

**Agent's Persona:** The agent greets the user by name, knows their current projects, remembers their writing preferences, retains important dates and temporal context from past sessions, and behaves like a trusted senior analyst colleague.

---

## 3. Core Functional Requirements

### 3.1 Session Initialisation & Personalised Greeting

- On every session start, the agent loads all four memory namespaces for the user.
- Simultaneously fires parallel RAG queries against the company DB to pre-fetch relevant context based on the user's active project/topic areas.
- Greets the user by name, references their current project context, and asks an opening question to understand today's goal.

### 3.2 Memory System (Four Namespaces)

| Namespace | Content | Updated When |
|---|---|---|
| `user_profile` | Name, role, seniority, communication style, personality notes | First session + whenever user reveals personal context |
| `company_profile` | Company name, industry, key competitors, strategic priorities, domain vocabulary | When new company info surfaces in research |
| `user_preferences` | Report format (PDF/MD/DOCX), verbosity, citation style, preferred frameworks (Porter's 5, SWOT, etc.) | When user edits outputs or gives stylistic feedback |
| `episodic_memory` | Date-stamped research notes: "On 2026-02-15, found that X competitor launched Y", key temporal anchors | After every research session, at report approval |

All memory updates use an LLM (Claude Haiku) with a targeted-update prompt. The LLM never overwrites existing memory — it makes surgical additions and corrections only.

### 3.3 Context Retrieval Layer

Before any planning or research, the agent:
1. Fires **parallel** RAG queries to Supabase (using the user's query + their current project context)
2. Fires **parallel** memory reads across all four namespaces
3. Combines retrieved chunks + memory into a `retrieved_context` state field
4. Uses this to inform the planner

### 3.4 Planner Agent

- Reads the user's query, retrieved context, and full memory profile
- Produces a structured research plan: topic breakdown, sub-questions, data sources (company DB vs web), output format
- Returns the plan as a structured Pydantic object with sections and rationale

### 3.5 HITL Gate 1 — Plan Review

- Shows the user the research plan in rich markdown via Agent Inbox interrupt
- User can:
  - **Accept** — proceed to research
  - **Edit** — modify plan (agent re-plans with edits as context)
  - **Respond** — give textual feedback (agent re-plans incorporating feedback)
  - **Ignore** — abort
- Memory write triggered: user's feedback on the plan updates `user_preferences` (e.g., preferred research depth, frameworks)

### 3.6 Research Subgraph

A fully autonomous multi-agent subgraph with:

#### Research Supervisor Node
- Receives the approved plan + retrieved context
- Decomposes the plan into N parallel research tasks
- Dispatches task agents using LangGraph `Send()` API
- Reviews returned findings from each agent: approves or sends back with specific critique
- After all tasks approved, synthesises discoveries into a structured summary
- Writes episodic memory entries for key findings

**Tools:** `rag_tool`, `web_search_tool`, `memory_search_tool`, `write_memory_tool`, `Question`

#### Task Agent Nodes (1..N, dynamically dispatched)
- Each agent receives: a specific sub-question, relevant context chunks
- Autonomously calls RAG and web search to answer its assigned question
- Returns structured findings: answer, sources, confidence, key quotes

**Tools:** `rag_tool`, `web_search_tool`

#### Supervisor Review Loop
- After each task agent returns, the supervisor evaluates:
  - Is the evidence sufficient?
  - Are sources credible?
  - Does it directly answer the sub-question?
- If insufficient → agent is re-dispatched with critique
- If approved → findings appended to `research_findings` state

### 3.7 HITL Gate 2 — Discovery Review

- Research supervisor presents top-level discoveries to the user
- Asks intelligent, contextual follow-up questions such as:
  - "We found X — do you want a deeper competitive breakdown here?"
  - "Competitor Y's strategy appears to contradict your firm's assumption Z — should we investigate?"
  - "We have strong data on A but limited data on B — shall we prioritise B?"
- User can:
  - **Respond** with specific follow-up directions → supervisor dispatches targeted agents
  - **Accept** → proceed to report writing
  - **Ignore** specific finding → supervisor notes it (memory write to `episodic_memory`)
- Memory write: preferences and notable decisions written to `user_preferences` and `episodic_memory`

This gate can loop: user keeps asking follow-ups, supervisor keeps dispatching agents, until user issues an approve command.

### 3.8 Report Writing Subgraph

A focused report-generation subgraph:

#### Report Supervisor Node
- Reads: full `research_findings`, user's `user_profile`, `user_preferences` (format, style, frameworks)
- Decomposes into report sections (Executive Summary, Market Analysis, Gap Analysis, Recommendations, etc.)
- Dispatches section-level write tasks to Writer Agents

**Tools:** `memory_search_tool`, `write_memory_tool`, `Question`

#### Writer Agent Node
- Receives a section assignment + all relevant context chunks
- Writes the section in the user's documented style
- Includes citations, data references, strategic frameworks as preferred

**Tools:** `rag_tool`, `memory_search_tool`

#### Report Assembly
- Supervisor combines approved sections into the final report
- Formats to user's preferred output (Markdown / structured JSON for DOCX/PDF generation)

### 3.9 HITL Gate 3 — Final Report Review

The most flexible gate. User can:

1. **Approve** → select output format → report saved to Supabase + chunked/embedded back into company DB
2. **Ask questions** → routed to Research Supervisor (which has full research context) for direct answers; conversation continues within the same thread
3. **Request re-research** → goes back to Research Supervisor with new direction (can fire new task agents)
4. **Request re-planning** → goes all the way back to Planner Agent with current context preserved
5. **Ignore** → session ends without saving

Memory write on approval: `episodic_memory` updated with session summary, key findings, date.

### 3.10 Report Storage

On final approval:
- Report stored in Supabase `reports` table with metadata: `user_id`, `session_id`, `date`, `topic_tags`, `format`, `project_name`
- Report chunked + embedded → inserted into Supabase `documents` table for future RAG retrieval by the agent or other consultants
- User is shown a confirmation with the report reference ID

---

## 4. Non-Functional Requirements

### 4.1 Persistence
- All conversation messages persist via LangGraph thread checkpointer (PostgreSQL in prod)
- Memory persists via LangGraph BaseStore (Supabase-backed in prod)
- Every session resumes with full context of previous sessions for the same user

### 4.2 Personalisation
- Agent must greet user by first name on every session start
- Agent must reference the user's active project if known from memory
- All LLM system prompts must be dynamically injected with memory content at call time

### 4.3 Performance
- Initial context load (RAG + memory) must complete before the first agent response
- Research task agents must run in parallel where possible
- Use Claude Haiku for: memory writes, quick routing decisions, structured output parsing
- Use Claude Sonnet for: planner, research supervisor, report writer

### 4.4 Observability
- LangSmith tracing enabled for all graph runs
- Each node logs its input state keys and output delta
- Memory reads/writes logged with namespace + content size

### 4.5 Security
- Supabase access uses service key (server-side only, never exposed)
- User isolation enforced at memory namespace level (all namespaces keyed by `user_id`)
- Report storage uses Row Level Security in Supabase

---

## 5. Data Model

### Supabase Tables Required

#### `documents` (company knowledge base)
```sql
id          uuid primary key
content     text
embedding   vector(1536)
metadata    jsonb  -- {source, document_type, project, date, tags}
created_at  timestamptz
```

#### `reports` (generated reports)
```sql
id           uuid primary key
user_id      text
session_id   text
title        text
content      text
format       text  -- markdown | json
topic_tags   text[]
project_name text
created_at   timestamptz
```

#### `report_chunks` (embedded report content for future RAG)
```sql
id          uuid primary key
report_id   uuid references reports(id)
content     text
embedding   vector(1536)
metadata    jsonb
created_at  timestamptz
```

### LangGraph BaseStore Memory Structure

```
("strategic_analyst", {user_id}, "user_profile")       → key: "profile"
("strategic_analyst", {user_id}, "company_profile")    → key: "profile"
("strategic_analyst", {user_id}, "user_preferences")   → key: "preferences"
("strategic_analyst", {user_id}, "episodic_memory")    → key: "episodes"
```

---

## 6. Agent State

```python
class AgentState(MessagesState):
    # Session identity
    user_id: str
    session_id: str

    # User's current request
    query: str

    # Context retrieval
    retrieved_context: list[dict]      # chunks from RAG
    memory_context: dict               # loaded memory namespaces

    # Planning
    research_plan: str                 # structured plan from planner
    plan_approved: bool

    # Research
    research_tasks: list[dict]         # decomposed tasks for task agents
    research_findings: dict            # approved findings per task
    supervisor_summary: str            # research supervisor's synthesis

    # Report
    report_sections: dict              # section_name → content
    report_draft: str                  # assembled draft
    final_report: str                  # approved final
    report_format: str                 # markdown | json | pdf

    # Routing
    current_phase: Literal["init", "planning", "research", "report", "final", "end"]
    hitl_response_type: str            # last HITL response type
```

---

## 7. Graph Structure

### Main Graph Nodes

```
START
  ↓
[context_loader]          ← parallel RAG + memory reads
  ↓
[greeting_node]           ← personalised greeting using memory
  ↓
[planner_agent]           ← produces structured research plan
  ↓
[hitl_plan_gate]          ← HITL Gate 1
  ↓ (approve) or → [planner_agent] (feedback loop)
[research_subgraph]       ← full research pipeline
  ↓
[hitl_discovery_gate]     ← HITL Gate 2 (can loop back into research)
  ↓ (approve)
[report_subgraph]         ← report writing pipeline
  ↓
[hitl_final_gate]         ← HITL Gate 3 (multi-route)
  ↓ (approve)
[save_report_node]
  ↓
END
```

### Research Subgraph Nodes

```
START
  ↓
[research_supervisor]     ← receives plan, dispatches via Send()
  ↓ (parallel Send)
[task_agent_1..N]         ← independent research workers
  ↓ (all return to supervisor)
[supervisor_review]       ← approve or re-dispatch
  ↓ (all approved)
[discovery_synthesiser]   ← builds supervisor_summary
  ↓
END
```

### Report Subgraph Nodes

```
START
  ↓
[report_supervisor]       ← reads findings + user prefs, dispatches sections
  ↓ (parallel Send)
[writer_agent_1..N]       ← section writers
  ↓ (all return)
[section_reviewer]        ← approve or re-write
  ↓ (all approved)
[report_assembler]        ← combine into final draft
  ↓
END
```

---

## 8. HITL Interrupt Schema

All interrupts follow the Agent Inbox schema:

```python
{
    "action_request": {
        "action": "descriptive action string",
        "args": {}  # optional structured args for editing
    },
    "config": {
        "allow_ignore": True,
        "allow_respond": True,
        "allow_edit": True,
        "allow_accept": True,
    },
    "description": "# Markdown content shown to user"
}
```

---

## 9. Tools Specification

### `rag_tool`
- Searches Supabase `documents` table using pgvector cosine similarity
- Accepts: `query: str`, `top_k: int = 5`, `filter_metadata: dict = None`
- Returns: list of `{content, metadata, similarity_score}`

### `web_search_tool`
- Wraps Tavily Search API
- Accepts: `query: str`, `max_results: int = 5`
- Returns: list of `{url, title, content, score}`

### `memory_search_tool`
- Searches a specific memory namespace for relevant content
- Accepts: `namespace: str`, `query: str`
- Returns: matched memory content as string

### `write_memory_tool`
- Triggers an LLM-powered targeted memory update
- Accepts: `namespace: str`, `context_messages: list`, `update_reason: str`
- Internally: reads current memory → LLM produces updated version → writes back

### `Question` (signal tool)
- Pydantic class tool: `content: str`
- Used by supervisor to ask the user clarifying questions via HITL

### `Done` (signal tool)
- Pydantic class tool: `done: bool`
- Used to signal workflow completion

---

## 10. Prompt System

All prompts use XML tag structure for clarity:

```
<Role> ... </Role>
<Background> {user_profile} </Background>
<CompanyContext> {company_profile} </CompanyContext>
<Preferences> {user_preferences} </Preferences>
<EpisodicContext> {episodic_memory} </EpisodicContext>
<Instructions> ... </Instructions>
<Rules> ... </Rules>
```

Dynamic injection: every LLM call reads current memory from store and injects it into the system prompt at call time.

---

## 11. Implementation Phases

| Phase | Scope | Plan File |
|---|---|---|
| 0 | Architecture overview & design decisions | `plan/00_overview.md` |
| 1 | Project setup: deps, folder structure, pyproject, langgraph.json | `plan/01_project_setup.md` |
| 2 | State & schemas: AgentState, all Pydantic models | `plan/02_state_and_schemas.md` |
| 3 | Memory system: namespaces, get/write helpers, update LLM | `plan/03_memory_system.md` |
| 4 | Tools: RAG, web search, memory tools, signal tools | `plan/04_tools.md` |
| 5 | Main graph nodes: context loader, greeting, planner, save report | `plan/05_main_graph_nodes.md` |
| 6 | Research subgraph: supervisor, task agents, Send() dispatch | `plan/06_research_subgraph.md` |
| 7 | Report writing subgraph: supervisor, writer agents, assembler | `plan/07_report_subgraph.md` |
| 8 | HITL gates: all three interrupt nodes with routing logic | `plan/08_hitl_gates.md` |
| 9 | Prompts: all system prompts, memory update prompts | `plan/09_prompts.md` |
| 10 | langgraph.json, testing, LangSmith tracing setup | `plan/10_langgraph_config_and_testing.md` |

---

## 12. Success Criteria

- [ ] Agent greets user by name on session start
- [ ] Agent correctly loads all four memory namespaces before responding
- [ ] RAG retrieval returns relevant company DB chunks within 3 seconds
- [ ] Research plan HITL gate correctly loops back on feedback
- [ ] Research task agents run in parallel (verified via LangSmith trace)
- [ ] Research supervisor correctly rejects and re-dispatches insufficient findings
- [ ] Discovery gate loops correctly for follow-up research
- [ ] Report matches user's documented style preferences from memory
- [ ] Final report stored in Supabase with correct metadata
- [ ] Report chunks embedded back into company DB for future retrieval
- [ ] Memory is updated after every HITL interaction
- [ ] New session correctly recalls previous episodic memories
