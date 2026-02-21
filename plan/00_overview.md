# Plan Step 00 — Architecture Overview & Design Decisions

## Goal
Establish the full mental model of the system before writing any code. Every subsequent step references decisions made here.

---

## System Overview

```
User
  │
  ▼
┌─────────────────────────────────────────────────────┐
│               MAIN GRAPH (main_graph.py)            │
│                                                     │
│  [context_loader] ──────────────────────────────┐   │
│       ↓ (parallel RAG + memory reads)           │   │
│  [greeting_node]                                │   │
│       ↓                                         │   │
│  [planner_agent]  ◄──────────────────────────── │   │
│       ↓                                         │   │
│  [hitl_plan_gate] ──→ (feedback) → [planner]    │   │
│       ↓ (approved)                              │   │
│  ┌──────────────────────────────┐               │   │
│  │    RESEARCH SUBGRAPH         │               │   │
│  │  [research_supervisor]       │               │   │
│  │    ↓ Send() parallel         │               │   │
│  │  [task_agent_1..N]           │               │   │
│  │    ↓ return to supervisor    │               │   │
│  │  [supervisor_review]         │               │   │
│  │    ↓ (all approved)          │               │   │
│  │  [discovery_synthesiser]     │               │   │
│  └──────────────────────────────┘               │   │
│       ↓                                         │   │
│  [hitl_discovery_gate] ──→ (follow-up) → [research] │
│       ↓ (approved)                              │   │
│  ┌──────────────────────────────┐               │   │
│  │    REPORT SUBGRAPH           │               │   │
│  │  [report_supervisor]         │               │   │
│  │    ↓ parallel Send()         │               │   │
│  │  [writer_agent_1..N]         │               │   │
│  │    ↓ return                  │               │   │
│  │  [section_reviewer]          │               │   │
│  │    ↓                         │               │   │
│  │  [report_assembler]          │               │   │
│  └──────────────────────────────┘               │   │
│       ↓                                         │   │
│  [hitl_final_gate]                              │   │
│    ↓ approve    ↓ re-research  ↓ re-plan        │   │
│  [save_report] [research_sub] [planner]         │   │
│       ↓                                         │   │
│      END                                        │   │
└─────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Decision 1: Single Shared State
All graphs and subgraphs share `AgentState(MessagesState)`. This avoids complex state mapping at subgraph boundaries and keeps the mental model simple. The research subgraph and report subgraph simply read from and write to the same fields.

### Decision 2: Memory is Always Loaded Before First Response
The `context_loader` node fires before the greeting. It runs parallel async calls:
- `asyncio.gather(load_rag_context(), load_all_memory_namespaces())`
This ensures every subsequent node has memory and context available in state.

### Decision 3: Command vs Conditional Edges
- Use `Command(goto=, update=)` at HITL gates and planner (routing depends on interrupt response)
- Use `add_conditional_edges` after research supervisor (routing depends on `should_continue` logic)

### Decision 4: Send() for Parallel Task Agents
The research supervisor uses `Send("task_agent", task_payload)` to dispatch N agents simultaneously. This is LangGraph's built-in fan-out pattern. The supervisor receives all results before proceeding.

### Decision 5: Memory Updates Are LLM-Driven
Every memory write goes through a Claude Haiku call that performs a targeted update. Raw overwrite is forbidden. This preserves all accumulated knowledge while integrating new information.

### Decision 6: Subgraphs Compiled Without Checkpointer
Only `main_graph.compile(checkpointer=checkpointer, store=store)` gets persistence.
Both `research_graph.compile()` and `report_graph.compile()` are compiled bare — they inherit from the parent at runtime.

### Decision 7: Three HITL Gates, Each Has All Four Response Types
Every interrupt node supports: `accept`, `edit`, `respond`, `ignore`.
- `accept` → proceed along happy path
- `edit` → modify the proposed action/plan/report
- `respond` → text feedback → re-process with feedback
- `ignore` → skip or abort

### Decision 8: User ID is Part of Every Thread Config
```python
thread_config = {
    "configurable": {
        "thread_id": session_id,
        "user_id": user_id
    }
}
```
The `user_id` flows into memory namespace keys so each user's memory is isolated.

---

## File Creation Order (for safe implementation)

1. `pyproject.toml` + `langgraph.json` → environment boots
2. `schemas.py` → state is defined, everything else can import it
3. `memory.py` → memory helpers, no other local deps
4. `tools/` → each tool is standalone
5. `prompts.py` → no local deps
6. `nodes/context_loader.py` → depends on tools + memory
7. `nodes/planner.py` → depends on tools + prompts
8. `nodes/hitl_gates.py` → depends on schemas
9. `nodes/memory_writer.py` → depends on memory.py
10. `nodes/report_saver.py` → depends on tools (Supabase client)
11. `subgraphs/research/` → depends on everything above
12. `subgraphs/report/` → depends on everything above
13. `main_graph.py` → wires it all together
