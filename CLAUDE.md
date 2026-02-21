# CLAUDE.md — Strategic Analyst Ambient RAG Agent

## Project Identity

**Name:** `strategic-analyst-agent`
**Purpose:** A personalised, ambient LangGraph agent that serves as a senior strategic analyst for consultants. It retrieves from the company's Supabase vector database, remembers the user's working style and episodic research history, conducts multi-agent research with human-in-the-loop gates, and produces detailed strategic reports.

**Reference Architecture:** `LANGGRAPH_AMBIENT_AGENT_REFERENCE.md` — treat this as the canonical LangGraph patterns guide. All graph construction, state, HITL, memory, and tool patterns come from there.

---

## Stack Decisions (Non-Negotiable)

| Concern | Choice |
|---|---|
| Agent framework | LangGraph (`langgraph>=0.2`) |
| LLM | Claude (claude-sonnet-4-6 primary, claude-haiku-4-5 for cheap utility calls) |
| LLM SDK | `langchain-anthropic` |
| Vector DB / Company DB | Supabase (pgvector) |
| Web Search | Tavily (`langchain-community` TavilySearch) |
| Memory persistence | LangGraph `BaseStore` (`InMemoryStore` for dev, Supabase-backed for prod) |
| Graph state persistence | LangGraph `MemorySaver` (dev) / PostgreSQL checkpointer (prod) |
| Package manager | `uv` |
| Python version | `>=3.12` |
| Config / secrets | `python-dotenv`, `.env` file |

---

## File & Folder Conventions

```
src/
  strategic_analyst/
    __init__.py
    main_graph.py          ← top-level compiled graph (entry point)
    schemas.py             ← ALL state, input, output, Pydantic models
    prompts.py             ← ALL system prompts and default memory strings
    memory.py              ← get_memory(), update_memory(), write_memory()
    configuration.py       ← RunnableConfig helpers
    nodes/
      context_loader.py    ← parallel RAG + memory load on session start
      planner.py           ← research plan LLM node
      hitl_gates.py        ← all interrupt() nodes
      memory_writer.py     ← LLM-driven memory update node
      report_saver.py      ← store final report to Supabase
    tools/
      base.py              ← get_tools(), get_tools_by_name() registry
      rag_tool.py          ← Supabase pgvector retrieval
      web_search_tool.py   ← Tavily wrapper
      memory_tools.py      ← memory_search_tool, write_memory_tool
      question_tool.py     ← Question, Done signal tools
    subgraphs/
      research/
        research_graph.py  ← compiled research subgraph
        supervisor.py      ← research supervisor node
        task_agent.py      ← individual research worker node (reusable)
      report/
        report_graph.py    ← compiled report-writing subgraph
        supervisor.py      ← report supervisor node
        writer_agent.py    ← writer worker node
langgraph.json
pyproject.toml
.env
.env.example
PRD.md
plan/
  00_overview.md
  01_project_setup.md
  02_state_and_schemas.md
  03_memory_system.md
  04_tools.md
  05_main_graph_nodes.md
  06_research_subgraph.md
  07_report_subgraph.md
  08_hitl_gates.md
  09_prompts.md
  10_langgraph_config_and_testing.md
```

---

## Architecture Rules

### 1. Graph Compilation
- Only the **outermost** `main_graph.py` graph is compiled with `checkpointer` and `store`.
- Research and report subgraphs are compiled **without** checkpointer/store — they inherit at runtime.

### 2. State
- A single `AgentState(MessagesState)` extends LangGraph's `MessagesState`.
- All subgraphs share the same `AgentState` — no separate state classes per subgraph unless strictly needed.
- Never mutate state objects — use `.model_copy(update={...})`.

### 3. Memory Namespaces

| Namespace | Key | Purpose |
|---|---|---|
| `("strategic_analyst", user_id, "user_profile")` | `"profile"` | Name, role, company, communication style |
| `("strategic_analyst", user_id, "company_profile")` | `"profile"` | Company context, industry, domain |
| `("strategic_analyst", user_id, "user_preferences")` | `"preferences"` | Research style, report format, verbosity |
| `("strategic_analyst", user_id, "episodic_memory")` | `"episodes"` | Important dates, temporal notes per session |

- All memory reads inject content into system prompts at runtime.
- All memory writes use an LLM (haiku) with targeted-update prompt — **never overwrite the whole profile**.

### 4. HITL Gates
Three mandatory interrupt points:
1. **Plan Gate** — after planner produces a research plan
2. **Discovery Gate** — after research supervisor summarises findings (can loop back)
3. **Final Report Gate** — before saving; user can re-research, re-plan, or approve

Each interrupt uses the Agent Inbox schema: `{action_request, config, description}`.

### 5. Tools Per Agent

| Agent | Tools |
|---|---|
| Main orchestrator / context loader | `rag_tool`, `memory_search_tool` |
| Planner | `rag_tool`, `memory_search_tool` |
| Research Supervisor | `rag_tool`, `web_search_tool`, `memory_search_tool`, `write_memory_tool`, `Question` |
| Research Task Agents | `rag_tool`, `web_search_tool` |
| Report Supervisor | `memory_search_tool`, `write_memory_tool`, `Question` |
| Report Writer Agent | `rag_tool`, `memory_search_tool` |

### 6. Routing
- Use `Command(goto=, update=)` for nodes that both update state and choose destination.
- Use `add_conditional_edges` for `should_continue` style routing after LLM calls.

### 7. Parallelism
- Initial context load: fire RAG + memory reads in parallel using `asyncio.gather` inside the node (not separate graph nodes).
- Research task agents: supervisor dispatches multiple agents that can run in parallel via LangGraph's `Send()` API.

### 8. Report Storage
- On approval, the report is stored in Supabase with metadata tags: `user_id`, `session_id`, `date`, `topic`, `format`.
- The report is also chunked and embedded back into the company database for future RAG retrieval.

---

## Coding Standards

- All node functions must have explicit type hints: `def node(state: AgentState, store: BaseStore) -> Command[...]`
- Use `async` for all Supabase and external API calls.
- Prompts live in `prompts.py` only — no inline prompt strings in node files.
- Tool definitions use `@tool` decorator or Pydantic class pattern (see reference doc Section 8).
- Structured outputs use `llm.with_structured_output(PydanticModel)`.
- Import LangGraph primitives at top: `from langgraph.graph import StateGraph, START, END`, `from langgraph.types import interrupt, Command`.

---

## Environment Variables

See `.env.example` for the full list. Required at minimum:
- `ANTHROPIC_API_KEY`
- `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_DB_URL`
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY` (for tracing)

---

## Running Locally

```bash
# Install deps
uv sync

# Start LangGraph dev server (reads langgraph.json)
langgraph dev

# Or run directly
uv run python -m strategic_analyst
```

---

## Testing

- Use `MemorySaver` + `InMemoryStore` in all tests.
- Test each subgraph in isolation before testing the full graph.
- Use `thread_config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": "test_user"}}`.
- Resume interrupted graphs with `Command(resume=[...])`.

---

## DO NOT

- Do not put prompts inline in node functions.
- Do not compile subgraphs with checkpointer or store.
- Do not mutate state objects in place.
- Do not use `gpt-*` models — this project uses Claude exclusively.
- Do not use `tool_choice="any"` where `tool_choice="required"` is appropriate.
- Do not skip memory writes after HITL interactions — every feedback loop is a learning opportunity.
