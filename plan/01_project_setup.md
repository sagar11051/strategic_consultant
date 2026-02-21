# Plan Step 01 — Project Setup

## Goal
Bootstrap the project: folder structure, `pyproject.toml`, `langgraph.json`, and package initialisation. After this step the dev server starts and imports resolve.

---

## 1.1 Folder Structure to Create

```
basic_rag_agent/
├── CLAUDE.md
├── PRD.md
├── plan/
├── .env
├── .env.example
├── langgraph.json
├── pyproject.toml
├── .python-version           ← already exists (3.12)
└── src/
    └── strategic_analyst/
        ├── __init__.py
        ├── main_graph.py
        ├── schemas.py
        ├── prompts.py
        ├── memory.py
        ├── configuration.py
        ├── nodes/
        │   ├── __init__.py
        │   ├── context_loader.py
        │   ├── planner.py
        │   ├── hitl_gates.py
        │   ├── memory_writer.py
        │   └── report_saver.py
        ├── tools/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── rag_tool.py
        │   ├── web_search_tool.py
        │   ├── memory_tools.py
        │   └── question_tool.py
        └── subgraphs/
            ├── __init__.py
            ├── research/
            │   ├── __init__.py
            │   ├── research_graph.py
            │   ├── supervisor.py
            │   └── task_agent.py
            └── report/
                ├── __init__.py
                ├── report_graph.py
                ├── supervisor.py
                └── writer_agent.py
```

---

## 1.2 `pyproject.toml`

```toml
[project]
name = "strategic-analyst-agent"
version = "0.1.0"
description = "Personalised ambient strategic analyst agent built on LangGraph"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # LangGraph core
    "langgraph>=0.2.0",
    "langgraph-cli[inmem]>=0.4.0",
    "langgraph-checkpoint-postgres>=3.0.4",

    # LangChain ecosystem
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-community>=0.3.0",
    "langsmith>=0.2.0",

    # Supabase
    "supabase>=2.4.0",
    "vecs>=0.4.0",           # optional: pgvector Python client
    "psycopg[binary]>=3.1.0",

    # Web search
    "tavily-python>=0.3.0",

    # Utilities
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "aiohttp>=3.9.0",

    # Report generation (optional, for DOCX/PDF export)
    "markdown>=3.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-env>=1.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/strategic_analyst"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## 1.3 `langgraph.json`

```json
{
    "dockerfile_lines": [],
    "graphs": {
        "strategic_analyst": "./src/strategic_analyst/main_graph.py:graph"
    },
    "python_version": "3.12",
    "env": ".env",
    "dependencies": ["."]
}
```

- `graph` is the name of the compiled top-level graph variable in `main_graph.py`
- `"dependencies": ["."]` → installs this package from `pyproject.toml`

---

## 1.4 `src/strategic_analyst/__init__.py`

Expose the compiled graph for LangGraph to import:

```python
from strategic_analyst.main_graph import graph

__all__ = ["graph"]
```

---

## 1.5 `src/strategic_analyst/configuration.py`

Holds the `RunnableConfig`-compatible configuration helper used across nodes:

```python
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class AgentConfiguration:
    """Runtime configuration injected via thread config."""
    user_id: str = "default_user"
    session_id: str = ""
    model_name: str = "claude-sonnet-4-6"
    utility_model_name: str = "claude-haiku-4-5-20251001"
    rag_top_k: int = 8
    max_research_tasks: int = 5
    max_supervisor_retries: int = 2

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "AgentConfiguration":
        configurable = (config or {}).get("configurable", {})
        return cls(**{k: v for k, v in configurable.items() if k in cls.__dataclass_fields__})
```

Usage in any node:
```python
def my_node(state: AgentState, config: RunnableConfig):
    cfg = AgentConfiguration.from_runnable_config(config)
    user_id = cfg.user_id
```

---

## 1.6 Setup Commands

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Verify langgraph CLI is available
uv run langgraph --version

# Start dev server (requires .env to be populated)
uv run langgraph dev
```

---

## Completion Checklist

- [ ] All directories created
- [ ] `pyproject.toml` written with full deps
- [ ] `langgraph.json` written
- [ ] `__init__.py` files created for all packages
- [ ] `configuration.py` written
- [ ] `uv sync` runs without errors
- [ ] `langgraph dev` starts (may error on missing .env — that's OK at this stage)
