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

**Identity fields in `AgentState`** (populated by `context_loader` from memory on session start):

| Field | Type | Source | Purpose |
|---|---|---|---|
| `user_id` | `str` | caller / `AgentInput` | Memory namespace key, Supabase filter |
| `session_id` | `str` | caller or auto-generated | Thread ID for checkpointer |
| `user_name` | `str` | `user_profile` memory | Personalised greetings and report headings |
| `user_role` | `str` | `user_profile` memory | Context for planner and prompts |
| `company_name` | `str` | `company_profile` memory | Report metadata, Supabase record |

`user_name`, `user_role`, `company_name` default to `""` until `context_loader` populates them.
`AgentInput` accepts optional hints for all three so the caller can seed them on first session.

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

## Database Schema

> All tables live in the Supabase project. The `chunks` and `documents` tables already exist.
> Run the SQL blocks below in the **Supabase SQL editor** to create the missing tables and functions.
> The LangGraph checkpointer tables are created automatically — see section 4.

---

### 1. Existing Tables (already in Supabase — do not recreate)

#### `documents`
Holds the parent document record for each ingested file.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | auto-generated |
| `title` | text | document title |
| `source_type` | text | e.g. `"pdf"`, `"url"`, `"notion"` |
| `source_path` | text | file path or URL |
| `file_type` | text | MIME or extension |
| `total_pages` | int4 | page count |
| `total_chunks` | int4 | chunk count produced |
| `metadata` | jsonb | arbitrary extra metadata |
| `created_at` | timestamptz | auto |
| `updated_at` | timestamptz | auto |

#### `chunks`
Holds individual text chunks with embeddings for vector search.

| Column | Type | Notes |
|---|---|---|
| `id` | uuid PK | auto-generated |
| `document_id` | uuid FK → documents.id | parent document |
| `content` | text | chunk text |
| `embedding` | vector(1024) | Voyage-3 embedding |
| `chunk_index` | int4 | position within document |
| `page_number` | int4 | source page (nullable) |
| `heading` | text | section heading (nullable) |
| `subheading` | text | subsection heading (nullable) |
| `section_path` | text[] | breadcrumb path of headings |
| `chunk_type` | text | e.g. `"paragraph"`, `"table"`, `"heading"` |
| `keywords` | text[] | extracted keywords |
| `questions` | jsonb | list of questions this chunk answers |
| `summary` | text | one-line chunk summary |
| `tsv` | tsvector | full-text search index |
| `created_at` | timestamptz | auto |

---

### 2. Tables to Create — Run in Supabase SQL Editor

#### `reports`
Stores completed, user-approved strategic reports.

```sql
CREATE TABLE IF NOT EXISTS reports (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         text NOT NULL,
    session_id      text NOT NULL,
    title           text NOT NULL,
    topic_tags      text[] NOT NULL DEFAULT '{}',
    project_name    text NOT NULL DEFAULT '',
    executive_summary text NOT NULL DEFAULT '',
    frameworks_used text[] NOT NULL DEFAULT '{}',
    content         text NOT NULL,               -- full report markdown
    format          text NOT NULL DEFAULT 'markdown', -- markdown | json | pdf
    metadata        jsonb NOT NULL DEFAULT '{}', -- arbitrary extra tags
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Index for fast per-user lookups
CREATE INDEX IF NOT EXISTS reports_user_id_idx ON reports (user_id);
CREATE INDEX IF NOT EXISTS reports_session_id_idx ON reports (session_id);
CREATE INDEX IF NOT EXISTS reports_created_at_idx ON reports (created_at DESC);

-- GIN index for topic tag filtering
CREATE INDEX IF NOT EXISTS reports_topic_tags_idx ON reports USING GIN (topic_tags);
```

#### `report_chunks`
Chunks of approved reports re-embedded for future RAG retrieval.
Uses the **same embedding model and dimension** as `chunks` (Voyage-3, 1024).

```sql
CREATE TABLE IF NOT EXISTS report_chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id       uuid NOT NULL REFERENCES reports (id) ON DELETE CASCADE,
    user_id         text NOT NULL,               -- denormalised for fast filter
    content         text NOT NULL,
    embedding       vector(1024),                -- Voyage-3, must match chunks table
    chunk_index     int4 NOT NULL,
    heading         text,
    keywords        text[] NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS report_chunks_report_id_idx ON report_chunks (report_id);
CREATE INDEX IF NOT EXISTS report_chunks_user_id_idx ON report_chunks (user_id);

-- HNSW vector index — same params as chunks table
CREATE INDEX IF NOT EXISTS report_chunks_embedding_idx
    ON report_chunks USING hnsw (embedding vector_cosine_ops);
```

---

### 3. Required RPC Functions — Run in Supabase SQL Editor

#### `match_documents` (company knowledge base search)
Called by `rag_tool.py`. Name must match `SUPABASE_MATCH_FUNCTION` env var.

```sql
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1024),
    match_count     int DEFAULT 8,
    filter          jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id          uuid,
    document_id uuid,
    content     text,
    heading     text,
    subheading  text,
    section_path text[],
    chunk_type  text,
    keywords    text[],
    summary     text,
    page_number int4,
    chunk_index int4,
    metadata    jsonb,
    similarity  float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        c.heading,
        c.subheading,
        c.section_path,
        c.chunk_type,
        c.keywords,
        c.summary,
        c.page_number,
        c.chunk_index,
        d.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE
        CASE
            WHEN filter != '{}' THEN d.metadata @> filter
            ELSE TRUE
        END
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

#### `match_report_chunks` (past reports search)
Called when agent retrieves from previously approved reports.

```sql
CREATE OR REPLACE FUNCTION match_report_chunks(
    query_embedding vector(1024),
    target_user_id  text,
    match_count     int DEFAULT 5
)
RETURNS TABLE (
    id          uuid,
    report_id   uuid,
    content     text,
    heading     text,
    keywords    text[],
    chunk_index int4,
    similarity  float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        rc.id,
        rc.report_id,
        rc.content,
        rc.heading,
        rc.keywords,
        rc.chunk_index,
        1 - (rc.embedding <=> query_embedding) AS similarity
    FROM report_chunks rc
    WHERE rc.user_id = target_user_id
    ORDER BY rc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

---

### 4. LangGraph Prod Checkpointer Tables (DO NOT create manually)

These are created automatically when you call `PostgresSaver.setup()` or
`AsyncPostgresSaver.setup()` using the `SUPABASE_DB_URL` connection string.
**Reference only — never create these by hand.**

```
checkpoints          → stores serialised graph state per (thread_id, checkpoint_ns, checkpoint_id)
checkpoint_blobs     → large blob data per channel/version
checkpoint_writes    → pending buffered writes per task
checkpoint_migrations → schema version tracking
```

In production `main_graph.py` init:
```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(os.getenv("SUPABASE_DB_URL")) as checkpointer:
    await checkpointer.setup()   # creates all four tables if they don't exist
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

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
