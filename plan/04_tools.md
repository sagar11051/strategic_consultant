# Plan Step 04 — Tools

## Goal
Implement all tools used by agent nodes. Each tool is standalone. The `base.py` registry provides `get_tools()` and `get_tools_by_name()` for consistent tool binding across nodes.

---

## 4.1 Tool Inventory

| Tool Name | File | Purpose | Used By |
|---|---|---|---|
| `rag_tool` | `rag_tool.py` | Supabase pgvector retrieval | context_loader, planner, task agents, writer agents |
| `web_search_tool` | `web_search_tool.py` | Tavily web search | task agents, research supervisor |
| `memory_search_tool` | `memory_tools.py` | Query user memory namespaces | supervisors, planner |
| `write_memory_tool` | `memory_tools.py` | Trigger LLM memory update | supervisors (via memory_writer node) |
| `Question` | `question_tool.py` | Signal tool to ask user a question | supervisors (triggers HITL) |
| `Done` | `question_tool.py` | Signal tool to indicate completion | all agents |

---

## 4.2 File: `src/strategic_analyst/tools/rag_tool.py`

```python
import os
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from supabase import create_client, Client
from langchain_anthropic import ChatAnthropic


class RAGInput(BaseModel):
    query: str = Field(description="The search query to retrieve relevant documents")
    top_k: int = Field(default=8, description="Number of results to return")
    filter_tags: Optional[list[str]] = Field(
        default=None,
        description="Optional list of metadata tags to filter by (e.g. ['market_analysis', 'Q4_2025'])"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Optional document type filter (e.g. 'report', 'research_note', 'competitor_analysis')"
    )


@tool(args_schema=RAGInput)
async def rag_tool(
    query: str,
    top_k: int = 8,
    filter_tags: Optional[list[str]] = None,
    document_type: Optional[str] = None,
) -> str:
    """
    Search the company's knowledge base for relevant documents, research, reports,
    and strategic intelligence. Use this to retrieve internal company data,
    previous research, competitor analyses, market data, and strategic documents.

    Returns a formatted string of the most relevant document chunks with their sources.
    """
    supabase: Client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

    # Generate embedding for the query
    # NOTE: We use Anthropic's embedding model OR we use a separate embedding service.
    # For Supabase pgvector, we need to call the embedding function first.
    # This will use the same embeddings model as the stored documents.
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Option A: Use Voyage AI embeddings (recommended for Supabase + Anthropic stack)
    # Option B: Use OpenAI text-embedding-3-small (commonly used with Supabase)
    # Placeholder — implement based on which embedding model was used to index docs

    # Call the Supabase match_documents RPC function (pgvector similarity search)
    # This assumes a stored procedure: match_documents(query_embedding, match_count, filter)
    params = {
        "query_embedding": query_embedding,   # populated after embedding generation
        "match_count": top_k,
    }

    if document_type:
        params["filter"] = {"document_type": document_type}

    response = supabase.rpc("match_documents", params).execute()

    if not response.data:
        return "No relevant documents found in the company knowledge base."

    # Format results
    formatted = []
    for i, doc in enumerate(response.data, 1):
        source = doc.get("metadata", {}).get("source", "Unknown source")
        doc_type = doc.get("metadata", {}).get("document_type", "document")
        similarity = doc.get("similarity", 0)
        content = doc.get("content", "")
        formatted.append(
            f"[{i}] Source: {source} | Type: {doc_type} | Relevance: {similarity:.2f}\n{content}\n"
        )

    return "\n---\n".join(formatted)
```

> **Implementation Note:** The embedding generation step depends on which embedding model was used to index the documents in Supabase. This will be filled in once the Supabase credentials and schema are provided. The tool structure above is complete; only the embedding call needs to be adapted.

---

## 4.3 File: `src/strategic_analyst/tools/web_search_tool.py`

```python
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query")
    max_results: int = Field(default=5, description="Maximum number of search results to return")
    search_depth: str = Field(
        default="advanced",
        description="Search depth: 'basic' for quick results, 'advanced' for deeper analysis"
    )


@tool(args_schema=WebSearchInput)
async def web_search_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced"
) -> str:
    """
    Search the web for current market intelligence, competitor information,
    industry trends, news, and publicly available strategic data.
    Use this when the company knowledge base doesn't have sufficient information
    or when you need current/real-time data.

    Returns formatted search results with title, URL, and content excerpt.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
    )

    results = []
    if response.get("answer"):
        results.append(f"**Quick Answer:** {response['answer']}\n")

    for i, result in enumerate(response.get("results", []), 1):
        results.append(
            f"[{i}] **{result.get('title', 'Untitled')}**\n"
            f"URL: {result.get('url', 'No URL')}\n"
            f"{result.get('content', 'No content')}\n"
        )

    return "\n---\n".join(results) if results else "No web search results found."
```

---

## 4.4 File: `src/strategic_analyst/tools/memory_tools.py`

```python
import os
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore


class MemorySearchInput(BaseModel):
    namespace_type: Literal[
        "user_profile", "company_profile", "user_preferences", "episodic_memory"
    ] = Field(description="Which memory namespace to search")
    query: str = Field(
        description="What you are looking for in this memory namespace"
    )


# NOTE: memory tools need access to `store` and `user_id` at runtime.
# These are injected via the node function signature, not through the tool directly.
# The tool is defined here but invoked inside nodes that have store access.
# See Section 4.6 for how to invoke these in nodes.

@tool(args_schema=MemorySearchInput)
def memory_search_tool(namespace_type: str, query: str) -> str:
    """
    Search the user's personal memory for relevant context.
    Use this to recall:
    - user_profile: who the user is, their role, communication style
    - company_profile: company context, industry, competitors
    - user_preferences: report format, preferred frameworks, verbosity
    - episodic_memory: past research sessions, important dates, temporal notes

    Always check memory before making assumptions about the user or their company.
    """
    # Implementation note: this tool requires store + user_id injection.
    # The actual store access happens in the node wrapper.
    # This docstring and schema are what the LLM sees for tool selection.
    raise NotImplementedError(
        "memory_search_tool must be invoked via the node-level wrapper that has store access"
    )


class WriteMemoryInput(BaseModel):
    namespace_type: Literal[
        "user_profile", "company_profile", "user_preferences", "episodic_memory"
    ] = Field(description="Which memory namespace to update")
    update_reason: str = Field(
        description="Explain what new information was discovered and why this memory should be updated"
    )
    context: str = Field(
        description="The new information or feedback that should be incorporated into memory"
    )


@tool(args_schema=WriteMemoryInput)
def write_memory_tool(namespace_type: str, update_reason: str, context: str) -> str:
    """
    Update a user memory namespace with new information discovered during research
    or revealed through user feedback.

    Use this when:
    - User reveals personal information (→ user_profile)
    - Research uncovers new company context (→ company_profile)
    - User gives style/format feedback (→ user_preferences)
    - A research session concludes with notable temporal findings (→ episodic_memory)

    IMPORTANT: This tool makes targeted updates only — it never overwrites existing memory.
    """
    raise NotImplementedError(
        "write_memory_tool must be invoked via the node-level wrapper that has store access"
    )
```

> **Implementation Note:** Memory tools need `store` and `user_id` to function. They are invoked inside node functions where those are available. The tool definitions above serve as the schema the LLM sees when deciding to call them. The actual execution happens in a custom `tool_node` wrapper inside each subgraph.

---

## 4.5 File: `src/strategic_analyst/tools/question_tool.py`

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
class Question(BaseModel):
    """
    Ask the user a clarifying question or request their input before proceeding.
    Use this when:
    - You need additional context to complete the research task
    - You found something interesting and want user direction on whether to pursue it
    - You are presenting discoveries and want to check if the user wants to go deeper
    - You need to confirm an assumption before writing a report section

    The question will be shown to the user via the human-in-the-loop interface.
    """
    content: str = Field(description="The question to ask the user")


@tool
class Done(BaseModel):
    """
    Signal that the current task or subtask is complete.
    Use this after:
    - Writing the final report section
    - Completing all assigned research tasks
    - Finishing a response to a user follow-up question

    Do NOT use this if you still have tool calls to make.
    """
    done: bool = Field(default=True, description="Set to True to signal completion")
    summary: str = Field(
        default="",
        description="Optional: brief summary of what was accomplished"
    )
```

---

## 4.6 File: `src/strategic_analyst/tools/base.py`

```python
from typing import Optional, List, Dict
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from strategic_analyst.tools.rag_tool import rag_tool
from strategic_analyst.tools.web_search_tool import web_search_tool
from strategic_analyst.tools.memory_tools import memory_search_tool, write_memory_tool
from strategic_analyst.tools.question_tool import Question, Done


# ── Tool Sets Per Agent Role ───────────────────────────────────────────────────

MAIN_AGENT_TOOLS = ["rag_tool", "memory_search_tool"]
PLANNER_TOOLS = ["rag_tool", "memory_search_tool"]
RESEARCH_SUPERVISOR_TOOLS = ["rag_tool", "web_search_tool", "memory_search_tool", "write_memory_tool", "Question"]
TASK_AGENT_TOOLS = ["rag_tool", "web_search_tool"]
REPORT_SUPERVISOR_TOOLS = ["memory_search_tool", "write_memory_tool", "Question"]
REPORT_WRITER_TOOLS = ["rag_tool", "memory_search_tool"]

ALL_TOOLS_REGISTRY = {
    "rag_tool": rag_tool,
    "web_search_tool": web_search_tool,
    "memory_search_tool": memory_search_tool,
    "write_memory_tool": write_memory_tool,
    "Question": Question,
    "Done": Done,
}


def get_tools(tool_names: Optional[List[str]] = None) -> List[BaseTool]:
    """Get a list of tool objects by name. Returns all if tool_names is None."""
    if tool_names is None:
        return list(ALL_TOOLS_REGISTRY.values())
    return [ALL_TOOLS_REGISTRY[name] for name in tool_names if name in ALL_TOOLS_REGISTRY]


def get_tools_by_name(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    """Convert a list of tools to a name-keyed dict for fast lookup."""
    return {tool.name: tool for tool in tools}


def make_tool_node(tools: List[BaseTool], store: BaseStore, user_id: str):
    """
    Create a tool execution node that handles store injection for memory tools.

    For normal tools (rag, web_search): invoke directly.
    For memory tools: inject store + user_id before invoking.
    """
    tools_by_name = get_tools_by_name(tools)

    from strategic_analyst.memory import (
        search_memory, update_memory_with_llm_async,
        user_profile_ns, company_profile_ns,
        user_preferences_ns, episodic_memory_ns, NAMESPACE_KEYS
    )

    NAMESPACE_MAP = {
        "user_profile": (user_profile_ns(user_id), NAMESPACE_KEYS["user_profile"]),
        "company_profile": (company_profile_ns(user_id), NAMESPACE_KEYS["company_profile"]),
        "user_preferences": (user_preferences_ns(user_id), NAMESPACE_KEYS["user_preferences"]),
        "episodic_memory": (episodic_memory_ns(user_id), NAMESPACE_KEYS["episodic_memory"]),
    }

    async def tool_node(state):
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            if tool_name == "memory_search_tool":
                ns_type = args.get("namespace_type", "user_profile")
                content = search_memory(store, user_id, ns_type, args.get("query", ""))
                results.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call["id"]
                })

            elif tool_name == "write_memory_tool":
                ns_type = args.get("namespace_type", "user_preferences")
                namespace, key = NAMESPACE_MAP[ns_type]
                await update_memory_with_llm_async(
                    store=store,
                    namespace=namespace,
                    key=key,
                    context_messages=[{"role": "user", "content": args.get("context", "")}],
                    update_reason=args.get("update_reason", "Agent-initiated memory update")
                )
                results.append({
                    "role": "tool",
                    "content": f"Memory namespace '{ns_type}' updated successfully.",
                    "tool_call_id": tool_call["id"]
                })

            elif tool_name in ["Question", "Done"]:
                # Signal tools — no execution, just acknowledgement
                results.append({
                    "role": "tool",
                    "content": f"Signal '{tool_name}' received.",
                    "tool_call_id": tool_call["id"]
                })

            else:
                # Regular tools (rag_tool, web_search_tool)
                tool_obj = tools_by_name[tool_name]
                observation = await tool_obj.ainvoke(args)
                results.append({
                    "role": "tool",
                    "content": str(observation),
                    "tool_call_id": tool_call["id"]
                })

        return {"messages": results}

    return tool_node
```

---

## Completion Checklist

- [ ] `rag_tool.py` written (embedding call stubbed, to be completed with Supabase schema)
- [ ] `web_search_tool.py` written and working (test with a sample query)
- [ ] `memory_tools.py` written with correct schemas
- [ ] `question_tool.py` written (Question and Done signal tools)
- [ ] `base.py` written with registry and `make_tool_node()` factory
- [ ] `tools/__init__.py` exports all tools
- [ ] Tool binding test: `llm.bind_tools(get_tools(TASK_AGENT_TOOLS))` works without error
