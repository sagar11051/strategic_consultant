# LangGraph Ambient Agent Building Reference

> A comprehensive technical reference extracted from a production email assistant agent. Use this document as an expert blueprint when building ambient agents with LangGraph.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core LangGraph Concepts](#2-core-langgraph-concepts)
3. [State Management](#3-state-management)
4. [Graph Construction Patterns](#4-graph-construction-patterns)
5. [Subgraph / Nested Agent Pattern](#5-subgraph--nested-agent-pattern)
6. [Command-Based Routing](#6-command-based-routing)
7. [Conditional Edges](#7-conditional-edges)
8. [Tool System](#8-tool-system)
9. [LLM Configuration & Structured Output](#9-llm-configuration--structured-output)
10. [Human-in-the-Loop (HITL)](#10-human-in-the-loop-hitl)
11. [Agent Inbox Integration](#11-agent-inbox-integration)
12. [Memory System (BaseStore)](#12-memory-system-basestore)
13. [Memory Update with LLM](#13-memory-update-with-llm)
14. [Checkpointers](#14-checkpointers)
15. [LangGraph Studio / LangSmith Compatibility](#15-langgraph-studio--langsmith-compatibility)
16. [langgraph.json Configuration](#16-langgraphjson-configuration)
17. [Cron Jobs & Scheduled Ingestion](#17-cron-jobs--scheduled-ingestion)
18. [Gmail Integration (Real External APIs)](#18-gmail-integration-real-external-apis)
19. [LangGraph SDK Client](#19-langgraph-sdk-client)
20. [Evaluation & Testing](#20-evaluation--testing)
21. [Prompt Engineering Patterns](#21-prompt-engineering-patterns)
22. [Project Structure](#22-project-structure)
23. [Dependencies](#23-dependencies)
24. [Key Patterns Summary](#24-key-patterns-summary)

---

## 1. Architecture Overview

The agent follows a **triage-then-act** pattern with progressive complexity layers:

```
Email Input
    |
    v
[Triage Router] --- LLM classifies email as: respond / notify / ignore
    |         \           \
    v          v           v
[Response    [Interrupt   [END]
 Agent]       Handler]
    |             |
    v             v
[LLM Call] <-> [Tool Execution / HITL Interrupt]
    |
    v
[Done] --> END
```

**Four progressive implementations exist:**
1. **Basic** (`email_assistant.py`) - Triage + response agent, no HITL
2. **HITL** (`email_assistant_hitl.py`) - Adds interrupt-based human review
3. **HITL + Memory** (`email_assistant_hitl_memory.py`) - Adds persistent memory via BaseStore
4. **HITL + Memory + Gmail** (`email_assistant_hitl_memory_gmail.py`) - Connects to real Gmail/Calendar APIs

---

## 2. Core LangGraph Concepts

### Imports You Always Need

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
```

### The Build Pattern

```python
# 1. Define state
# 2. Create StateGraph with state class
# 3. Add nodes (functions)
# 4. Add edges (connections)
# 5. Compile with checkpointer and/or store
```

---

## 3. State Management

### State Definition with MessagesState

```python
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState

# Input schema (what the graph receives)
class StateInput(TypedDict):
    email_input: dict

# Full state (extends MessagesState which has built-in `messages` key)
class State(MessagesState):
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]
```

**Key points:**
- `MessagesState` provides a built-in `messages` key with proper message append behavior
- `StateInput` restricts what the caller can pass in (used as `input=StateInput` in `StateGraph`)
- State updates are **merged** not replaced - returning `{"messages": [...]}` appends to existing messages
- The `messages` key uses a **reducer** that appends new messages to existing ones automatically

### Structured Output Schemas

```python
class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""
    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email"
    )

class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""
    chain_of_thought: str = Field(description="Reasoning about which preferences need update")
    user_preferences: str = Field(description="Updated user preferences")
```

---

## 4. Graph Construction Patterns

### Basic Graph

```python
# StateGraph accepts the state class and optionally an input schema
workflow = StateGraph(State, input=StateInput)

# Add nodes - can be functions or compiled subgraphs
workflow.add_node("triage_router", triage_router)
workflow.add_node("response_agent", compiled_subgraph)  # Nested graph!

# Add edges
workflow.add_edge(START, "triage_router")
# No explicit edge from triage_router needed - it uses Command for routing

# Compile
email_assistant = workflow.compile()
```

### Using the Fluent Builder API

```python
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)              # Function name becomes node name
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", agent)     # Explicit name for subgraph
    .add_edge(START, "triage_router")
    .add_edge("mark_as_read_node", END)
)
email_assistant = overall_workflow.compile()
```

---

## 5. Subgraph / Nested Agent Pattern

**This is one of the most important patterns.** The response agent is a separate compiled graph used as a node in the parent graph.

```python
# === INNER GRAPH (response agent) ===
agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"interrupt_handler": "interrupt_handler", END: END},
)

# Compile the inner agent (no checkpointer here - parent handles it)
response_agent = agent_builder.compile()

# === OUTER GRAPH (overall workflow) ===
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)  # <-- Subgraph as node!
    .add_edge(START, "triage_router")
)

# Checkpointer and store are passed ONLY at the top-level compile
email_assistant = overall_workflow.compile(checkpointer=checkpointer, store=store)
```

**Key rule:** Only compile the **outermost** graph with `checkpointer` and `store`. Inner subgraphs are compiled without them - they inherit from the parent at runtime.

---

## 6. Command-Based Routing

`Command` allows a node to **simultaneously update state AND choose the next node** to visit. This replaces the need for separate conditional edges in many cases.

```python
from langgraph.types import Command

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """The return type hint declares all possible destinations."""

    # ... LLM classification logic ...

    if classification == "respond":
        goto = "response_agent"
        update = {
            "classification_decision": "respond",
            "messages": [{"role": "user", "content": f"Respond to: {email_markdown}"}],
        }
    elif classification == "ignore":
        goto = END      # Use END constant, but in type hint use "__end__"
        update = {"classification_decision": "ignore"}
    elif classification == "notify":
        goto = "triage_interrupt_handler"
        update = {"classification_decision": "notify"}

    return Command(goto=goto, update=update)
```

**Key points:**
- Type hint `Command[Literal["node_a", "node_b", "__end__"]]` declares valid destinations
- `goto` picks the next node dynamically
- `update` merges into state (same as returning a dict from a normal node)
- `END` is used in code but `"__end__"` is used in the Literal type hint
- **No `add_edge` needed from this node** - Command handles routing

---

## 7. Conditional Edges

For nodes that return state (not Command), use conditional edges:

```python
def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route based on whether the LLM made tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"

# Wire it up
agent_builder.add_conditional_edges(
    "llm_call",           # Source node
    should_continue,      # Router function
    {                     # Mapping: return value -> node name
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)
```

**Conditional edge functions can also accept `store: BaseStore`** when memory is involved:

```python
def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "__end__"]:
    # ... can access store here if needed ...
```

---

## 8. Tool System

### Defining Tools with `@tool` Decorator

```python
from langchain_core.tools import tool
from pydantic import BaseModel

# Simple function tool
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

# Pydantic-class tool (for signal tools with no real execution)
@tool
class Done(BaseModel):
    """E-mail has been sent."""
    done: bool

@tool
class Question(BaseModel):
    """Question to ask user."""
    content: str
```

### Tool with Custom Input Schema (for complex args)

```python
from pydantic import BaseModel, Field

class SendEmailInput(BaseModel):
    email_id: str = Field(description="Gmail message ID to reply to")
    response_text: str = Field(description="Content of the reply")
    email_address: str = Field(description="Current user's email address")
    additional_recipients: Optional[List[str]] = Field(default=None, description="Optional CC recipients")

@tool(args_schema=SendEmailInput)
def send_email_tool(email_id: str, response_text: str, email_address: str, additional_recipients: Optional[List[str]] = None) -> str:
    """Send a reply to an existing email thread."""
    # ... implementation ...
```

### Tool Registry Pattern

```python
# base.py - Central tool registry
def get_tools(tool_names: Optional[List[str]] = None, include_gmail: bool = False) -> List[BaseTool]:
    """Get specified tools or all tools if tool_names is None."""
    from email_assistant.tools.default.email_tools import write_email, Done, Question
    from email_assistant.tools.default.calendar_tools import schedule_meeting, check_calendar_availability

    all_tools = {
        "write_email": write_email,
        "Done": Done,
        "Question": Question,
        "schedule_meeting": schedule_meeting,
        "check_calendar_availability": check_calendar_availability,
    }

    if include_gmail:
        from email_assistant.tools.gmail.gmail_tools import send_email_tool, check_calendar_tool, schedule_meeting_tool
        all_tools.update({...})

    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]

def get_tools_by_name(tools) -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    return {tool.name: tool for tool in tools}
```

### Binding Tools to LLM

```python
tools = get_tools()
tools_by_name = get_tools_by_name(tools)

# For structured routing output
llm_router = llm.with_structured_output(RouterSchema)

# For tool calling (force at least one tool call)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")
# Or allow optional tool calling:
llm_with_tools = llm.bind_tools(tools, tool_choice="any")
```

### Manual Tool Execution Node

```python
def tool_node(state: State):
    """Execute tool calls from the last AI message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({
            "role": "tool",
            "content": observation,
            "tool_call_id": tool_call["id"]
        })
    return {"messages": result}
```

---

## 9. LLM Configuration & Structured Output

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="model-name",
    api_key=os.getenv('API_KEY'),
    base_url="https://your-endpoint.com/v1",
    temperature=0.0
)

# Structured output (returns Pydantic object)
llm_router = llm.with_structured_output(RouterSchema)
result = llm_router.invoke([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
])
# result.classification, result.reasoning are typed fields

# Tool-bound LLM
llm_with_tools = llm.bind_tools(tools, tool_choice="required")
```

---

## 10. Human-in-the-Loop (HITL)

### The `interrupt()` Primitive

`interrupt()` **pauses graph execution** and sends data to an external client (like Agent Inbox). The graph stays suspended until someone resumes it with a response.

```python
from langgraph.types import interrupt

def interrupt_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    """Pause for human review of tool calls."""

    for tool_call in state["messages"][-1].tool_calls:

        # Some tools execute without interruption
        if tool_call["name"] not in ["write_email", "schedule_meeting", "Question"]:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue

        # Build the interrupt request
        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,     # User can edit tool args
                "allow_accept": True,   # User can approve as-is
            },
            "description": display_text,  # Markdown shown in Agent Inbox
        }

        # THIS LINE PAUSES THE GRAPH
        response = interrupt([request])[0]

        # Handle the 4 response types
        if response["type"] == "accept":
            # Execute tool with original args
            ...
        elif response["type"] == "edit":
            # Execute tool with edited args from response["args"]["args"]
            ...
        elif response["type"] == "ignore":
            # Skip tool, end workflow
            ...
        elif response["type"] == "response":
            # User provided text feedback in response["args"]
            ...
```

### HITL Response Types

| Type | What it means | How to handle |
|------|--------------|---------------|
| `accept` | User approves the action as-is | Execute tool with original args |
| `edit` | User modified the tool args | Execute tool with `response["args"]["args"]` |
| `ignore` | User wants to skip this action | Don't execute, often route to END |
| `response` | User provided text feedback | Feed feedback back to LLM for revision |

### Editing Tool Calls (State Immutability)

When a user edits tool call args, you must create a new copy of the AI message:

```python
elif response["type"] == "edit":
    edited_args = response["args"]["args"]
    ai_message = state["messages"][-1]
    current_id = tool_call["id"]

    # Create new list filtering out the edited call and adding updated version
    updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
        {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
    ]

    # IMPORTANT: Create a copy, don't mutate the original
    result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

    # Then execute the tool with edited args
    observation = tool.invoke(edited_args)
    result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
```

### Triage-Level Interrupts (Notification Handling)

For emails classified as "notify", a separate interrupt handler pauses to show the user the email and let them decide:

```python
def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Show notification email to user, let them decide to respond or ignore."""

    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,    # Can't edit a notification
            "allow_accept": False,  # Not an action to approve
        },
        "description": email_markdown,
    }

    response = interrupt([request])[0]

    if response["type"] == "response":
        # User wants to reply - route to response agent with their feedback
        goto = "response_agent"
    elif response["type"] == "ignore":
        # User dismisses the notification
        goto = END
```

---

## 11. Agent Inbox Integration

The interrupt system is designed to work with **Agent Inbox**, a web UI for reviewing agent actions remotely.

### Interrupt Request Schema

```python
request = {
    "action_request": {
        "action": "tool_name",          # or descriptive action string
        "args": {"key": "value"}        # tool arguments (editable in UI)
    },
    "config": {
        "allow_ignore": bool,   # Show "Ignore" button
        "allow_respond": bool,  # Show "Respond" text input
        "allow_edit": bool,     # Show "Edit" button (edit args JSON)
        "allow_accept": bool,   # Show "Accept" button
    },
    "description": "markdown string",  # Rich display in the inbox
}
```

### Formatting for Display

```python
def format_for_display(tool_call):
    """Format tool calls as readable markdown for Agent Inbox."""
    if tool_call["name"] == "write_email":
        return f"""# Email Draft
**To**: {tool_call["args"].get("to")}
**Subject**: {tool_call["args"].get("subject")}

{tool_call["args"].get("content")}
"""
    elif tool_call["name"] == "schedule_meeting":
        return f"""# Calendar Invite
**Meeting**: {tool_call["args"].get("subject")}
**Attendees**: {', '.join(tool_call["args"].get("attendees"))}
**Duration**: {tool_call["args"].get("duration_minutes")} minutes
"""
    elif tool_call["name"] == "Question":
        return f"""# Question for User
{tool_call["args"].get("content")}
"""
```

---

## 12. Memory System (BaseStore)

### How BaseStore Works

LangGraph's `BaseStore` is a **namespaced key-value store** that persists across graph invocations. It is separate from the checkpointer (which stores graph execution state).

```
Store Structure:
  namespace = ("email_assistant", "triage_preferences")
  key = "user_preferences"
  value = "string of preferences..."

  namespace = ("email_assistant", "response_preferences")
  key = "user_preferences"
  value = "string of preferences..."

  namespace = ("email_assistant", "cal_preferences")
  key = "user_preferences"
  value = "string of preferences..."
```

### Accessing Store in Node Functions

**Declare `store: BaseStore` as a parameter** in any node function that needs memory:

```python
from langgraph.store.base import BaseStore

def triage_router(state: State, store: BaseStore) -> Command[...]:
    """LangGraph auto-injects the store when the parameter is named `store`."""
    triage_instructions = get_memory(store, ("email_assistant", "triage_preferences"), default_triage_instructions)
    # ... use triage_instructions in prompt ...

def llm_call(state: State, store: BaseStore):
    cal_preferences = get_memory(store, ("email_assistant", "cal_preferences"), default_cal_preferences)
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)
    # ... inject into system prompt ...
```

### Memory Read Helper

```python
def get_memory(store, namespace, default_content=None):
    """Get memory from store, initialize with default if it doesn't exist."""
    # store.get(namespace_tuple, key) -> Item or None
    user_preferences = store.get(namespace, "user_preferences")

    if user_preferences:
        return user_preferences.value  # .value contains the stored data

    # First time: initialize with default
    store.put(namespace, "user_preferences", default_content)
    return default_content
```

### Memory Write

```python
store.put(namespace_tuple, key_string, value)
# Example:
store.put(("email_assistant", "response_preferences"), "user_preferences", "Be more formal")
```

### Three Memory Namespaces Used

| Namespace | Purpose | Updated When |
|-----------|---------|-------------|
| `("email_assistant", "triage_preferences")` | Rules for classifying emails | User ignores/responds to notify emails; user ignores drafted responses |
| `("email_assistant", "response_preferences")` | Email writing style/tone | User edits email drafts; user gives feedback on drafts |
| `("email_assistant", "cal_preferences")` | Calendar/meeting preferences | User edits meeting invites; user gives feedback on scheduling |

---

## 13. Memory Update with LLM

Memory updates are NOT simple overwrites. An LLM processes the feedback and makes **targeted updates** to the existing memory profile.

```python
def update_memory(store, namespace, messages):
    """Use an LLM to selectively update the memory profile."""

    # Get current memory
    user_preferences = store.get(namespace, "user_preferences")

    # Use structured output LLM to produce updated preferences
    llm_memory = ChatOpenAI(...).with_structured_output(UserPreferences)

    result = llm_memory.invoke([
        {"role": "system", "content": MEMORY_UPDATE_INSTRUCTIONS.format(
            current_profile=user_preferences.value,
            namespace=namespace
        )},
    ] + messages)  # messages contain the feedback context

    # Save updated memory
    store.put(namespace, "user_preferences", result.user_preferences)
```

### Memory Update Prompt (Critical)

```
# Instructions
- NEVER overwrite the entire memory profile
- ONLY make targeted additions of new information
- ONLY update specific facts that are directly contradicted by feedback
- PRESERVE all other existing information in the profile
- Format the profile consistently with the original style

# Reasoning Steps
1. Analyze the current memory profile structure and content
2. Review feedback messages from human-in-the-loop interactions
3. Extract relevant user preferences from these feedback messages
4. Compare new information against existing profile
5. Identify only specific facts to add or update
6. Preserve all other existing information
7. Output the complete updated profile
```

### When Memory Gets Updated

Memory updates are triggered by HITL actions:

```python
# User EDITS an email draft -> update response_preferences
if response["type"] == "edit" and tool_call["name"] == "write_email":
    update_memory(store, ("email_assistant", "response_preferences"), [{
        "role": "user",
        "content": f"User edited the email. Initial: {initial_args}. Edited: {edited_args}."
    }])

# User IGNORES a drafted email -> update triage_preferences
if response["type"] == "ignore" and tool_call["name"] == "write_email":
    update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + [{
        "role": "user",
        "content": "User ignored the draft. Update triage to not classify similar emails as respond."
    }])

# User gives text FEEDBACK -> update relevant namespace
if response["type"] == "response":
    update_memory(store, ("email_assistant", "response_preferences"), [{
        "role": "user",
        "content": f"User feedback: {user_feedback}. Update response preferences accordingly."
    }])

# Triage notification: user responds -> update triage_preferences
if response["type"] == "response":  # in triage_interrupt_handler
    update_memory(store, ("email_assistant", "triage_preferences"), [{
        "role": "user",
        "content": "User decided to respond to this email. Update triage preferences."
    }] + messages)
```

---

## 14. Checkpointers

Checkpointers save the **execution state of the graph** (current node, messages, state values) so it can be resumed after an interrupt.

### In-Memory Checkpointer (for testing/development)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
email_assistant = overall_workflow.compile(checkpointer=checkpointer, store=store)
```

### PostgreSQL Checkpointer (for production)

Listed in dependencies: `langgraph-checkpoint-postgres>=3.0.4`

```python
# Used when deploying with LangGraph Platform
# Configured automatically via langgraph.json
```

### Compile Pattern with Both

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = MemorySaver()
store = InMemoryStore()

# Only the TOP-LEVEL graph gets checkpointer and store
email_assistant = overall_workflow.compile(
    checkpointer=checkpointer,
    store=store
)
```

### Thread Config (Required for Checkpointer)

```python
import uuid

thread_id = uuid.uuid4()
thread_config = {"configurable": {"thread_id": thread_id}}

# Invoke with config
result = email_assistant.invoke({"email_input": email_data}, config=thread_config)

# Stream with config
for chunk in email_assistant.stream({"email_input": email_data}, config=thread_config):
    print(chunk)

# Resume after interrupt with Command
for chunk in email_assistant.stream(
    Command(resume=[{"type": "accept", "args": ""}]),
    config=thread_config
):
    print(chunk)

# Get current state
state = email_assistant.get_state(thread_config)
```

---

## 15. LangGraph Studio / LangSmith Compatibility

### LangSmith Tracing

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "E-mail Tool Calling and Response Evaluation"
```

### LangSmith Testing Integration

```python
from langsmith import testing as t

# Log inputs for tracing
t.log_inputs({"module": AGENT_MODULE, "test": "test_name"})

# Log outputs for tracing
t.log_outputs({
    "extracted_tool_calls": extracted_tool_calls,
    "response": all_messages_str
})
```

### LangSmith Dataset & Evaluation

```python
from langsmith import Client

client = Client()
dataset_name = "E-mail Triage Dataset"

# Create dataset
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name, description="...")
    client.create_examples(dataset_id=dataset.id, examples=examples_triage)

# Run evaluation
experiment_results = client.evaluate(
    target_function,
    data=dataset_name,
    evaluators=[classification_evaluator],
    experiment_prefix="E-mail assistant workflow",
    max_concurrency=2,
)
```

### Running in LangGraph Studio

LangGraph Studio reads `langgraph.json` and serves your graphs locally with a UI. Run:

```bash
langgraph dev
# or for production:
langgraph up
```

The studio automatically provides:
- Visual graph viewer
- Thread management
- Interrupt/resume UI
- State inspection
- Store inspection

---

## 16. langgraph.json Configuration

This file tells LangGraph Platform (and Studio) how to serve your graphs:

```json
{
    "dockerfile_lines": [],
    "graphs": {
        "langgraph101": "./src/email_assistant/langgraph_101.py:app",
        "email_assistant": "./src/email_assistant/email_assistant.py:email_assistant",
        "email_assistant_hitl": "./src/email_assistant/email_assistant_hitl.py:email_assistant",
        "email_assistant_hitl_memory": "./src/email_assistant/email_assistant_hitl_memory.py:email_assistant",
        "email_assistant_hitl_memory_gmail": "./src/email_assistant/email_assistant_hitl_memory_gmail.py:email_assistant",
        "cron": "./src/email_assistant/cron.py:graph"
    },
    "python_version": "3.11",
    "env": ".env",
    "dependencies": ["."]
}
```

**Format:** `"graph_name": "./path/to/file.py:variable_name"`

- `graph_name` - how it appears in Studio and API routes
- `variable_name` - the compiled graph object in the Python file
- `env` - path to .env file for environment variables
- `dependencies` - `["."]` means install the current package (from pyproject.toml)

---

## 17. Cron Jobs & Scheduled Ingestion

### Cron Graph Definition

A cron job is itself a LangGraph graph:

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph

@dataclass(kw_only=True)
class JobKickoff:
    """State for the cron job."""
    email: str
    minutes_since: int = 60
    graph_name: str = "email_assistant_hitl_memory_gmail"
    url: str = "http://127.0.0.1:2024"

async def main(state: JobKickoff):
    """Run the email ingestion process."""
    result = await fetch_and_process_emails(args)
    return {"status": "success" if result == 0 else "error"}

graph = StateGraph(JobKickoff)
graph.add_node("ingest_emails", main)
graph.set_entry_point("ingest_emails")
graph = graph.compile()
```

### Setting Up Cron via LangGraph SDK

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")

cron = await client.crons.create(
    "cron",                          # Graph name in langgraph.json
    schedule="*/10 * * * *",         # Every 10 minutes
    input={
        "email": "user@gmail.com",
        "minutes_since": 60,
        "graph_name": "email_assistant_hitl_memory_gmail",
        "url": "http://127.0.0.1:2024",
    }
)
```

---

## 18. Gmail Integration (Real External APIs)

### Credential Management

Credentials are loaded from multiple sources in priority order:

1. Directly passed parameters
2. Environment variables (`GMAIL_TOKEN`, `GMAIL_SECRET`)
3. Local files (`.secrets/token.json`, `.secrets/secrets.json`)

```python
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def get_credentials(gmail_token=None, gmail_secret=None):
    """Load OAuth2 credentials from token data."""
    # ... load token_data from env or file ...
    credentials = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes")
    )
    return credentials
```

### OAuth Setup Flow

```python
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar'
]

flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
credentials = flow.run_local_server(port=0)
# Save token_data to .secrets/token.json
```

### Email Ingestion to LangGraph Server

```python
from langgraph_sdk import get_client
import uuid, hashlib

async def ingest_email_to_langgraph(email_data, graph_name, url="http://127.0.0.1:2024"):
    client = get_client(url=url)

    # Create deterministic thread ID from Gmail thread ID
    raw_thread_id = email_data["thread_id"]
    thread_id = str(uuid.UUID(hex=hashlib.md5(raw_thread_id.encode("UTF-8")).hexdigest()))

    # Create or get thread
    try:
        thread_info = await client.threads.get(thread_id)
    except:
        thread_info = await client.threads.create(thread_id=thread_id)

    # Create a run
    run = await client.runs.create(
        thread_id,
        graph_name,
        input={"email_input": {
            "from": email_data["from_email"],
            "to": email_data["to_email"],
            "subject": email_data["subject"],
            "body": email_data["page_content"],
            "id": email_data["id"]
        }},
        multitask_strategy="rollback",  # Cancel previous runs on same thread
    )
```

### Mark as Read (Post-Processing)

```python
def mark_as_read(message_id, gmail_token=None, gmail_secret=None):
    """Remove UNREAD label after processing."""
    creds = get_credentials(gmail_token, gmail_secret)
    service = build("gmail", "v1", credentials=creds)
    service.users().messages().modify(
        userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()
```

In the Gmail workflow, `mark_as_read_node` runs after the `Done` tool:

```python
def should_continue(state, store) -> Literal["interrupt_handler", "mark_as_read_node"]:
    if tool_call["name"] == "Done":
        return "mark_as_read_node"  # Instead of END
    else:
        return "interrupt_handler"

agent_builder.add_node("mark_as_read_node", mark_as_read_node)
agent_builder.add_edge("mark_as_read_node", END)
```

---

## 19. LangGraph SDK Client

The SDK client communicates with a running LangGraph server:

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")

# Thread operations
thread = await client.threads.create(thread_id=thread_id)
thread = await client.threads.get(thread_id)
await client.threads.update(thread_id, metadata={...})

# Run operations
run = await client.runs.create(thread_id, graph_name, input={...}, multitask_strategy="rollback")
runs = await client.runs.list(thread_id)
await client.runs.delete(thread_id, run_id)

# Cron operations
cron = await client.crons.create("graph_name", schedule="*/10 * * * *", input={...})
```

---

## 20. Evaluation & Testing

### Test Setup with Dynamic Module Loading

```python
import importlib

# conftest.py allows switching implementations via CLI
def pytest_addoption(parser):
    parser.addoption("--agent-module", action="store", default="email_assistant_hitl_memory")

@pytest.fixture(scope="session")
def agent_module_name(request):
    return request.config.getoption("--agent-module")
```

### Setup Pattern for Tests

```python
def setup_assistant():
    checkpointer = MemorySaver()
    store = InMemoryStore()
    thread_id = uuid.uuid4()
    thread_config = {"configurable": {"thread_id": thread_id}}

    if AGENT_MODULE == "email_assistant_hitl_memory":
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer, store=store)
    elif AGENT_MODULE == "email_assistant_hitl":
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
    else:
        email_assistant = agent_module.overall_workflow.compile(checkpointer=checkpointer)
        store = None

    return email_assistant, thread_config, store
```

### Two Types of Evaluations

**1. Tool Call Evaluation** - Check that the right tools were called:

```python
@pytest.mark.langsmith(output_keys=["expected_calls"])
@pytest.mark.parametrize("email_input,email_name,criteria,expected_calls", create_response_test_cases())
def test_email_dataset_tool_calls(email_input, email_name, criteria, expected_calls):
    email_assistant, thread_config, _ = setup_assistant()
    result = email_assistant.invoke({"email_input": email_input}, config=thread_config)
    state = email_assistant.get_state(thread_config)
    extracted_tool_calls = extract_tool_calls(state.values["messages"])
    missing_calls = [call for call in expected_calls if call.lower() not in extracted_tool_calls]
    assert len(missing_calls) == 0
```

**2. LLM-as-Judge Criteria Evaluation** - Use an LLM to judge response quality:

```python
@pytest.mark.langsmith(output_keys=["criteria"])
def test_response_criteria_evaluation(email_input, email_name, criteria, expected_calls):
    # ... run agent ...
    eval_result = criteria_eval_structured_llm.invoke([
        {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Criteria: {criteria}\nResponse: {all_messages_str}"}
    ])
    assert eval_result.grade
```

### Running Tests

```bash
# Run all tests
python tests/run_all_tests.py --all

# Run specific implementation
python tests/run_all_tests.py --implementation email_assistant
```

---

## 21. Prompt Engineering Patterns

### System Prompt Structure

Prompts use a consistent XML-tag structure:

```python
triage_system_prompt = """
< Role >
Your role is to triage incoming emails based upon instructions and background information below.
</ Role >

< Background >
{background}
</ Background >

< Instructions >
Categorize each email into one of three categories:
1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that doesn't require a response
3. RESPOND - Emails that need a direct response
</ Instructions >

< Rules >
{triage_instructions}
</ Rules >
"""
```

### Agent System Prompt with Dynamic Injection

```python
agent_system_prompt_hitl_memory = """
< Role >
You are a top-notch executive assistant.
</ Role >

< Tools >
{tools_prompt}           # Injected tool descriptions
</ Tools >

< Instructions >
1. Carefully analyze the email content and purpose
2. IMPORTANT --- always call a tool and call one tool at a time
3. If the incoming email asks a direct question and you lack context, use the Question tool
4. For responding, use write_email tool
5. For meeting requests, use check_calendar_availability then schedule_meeting
   - Today's date is """ + datetime.now().strftime("%Y-%m-%d") + """
6. After using write_email, the task is complete
7. Use Done tool to indicate completion
</ Instructions >

< Background >
{background}             # User's personal context
</ Background >

< Response Preferences >
{response_preferences}   # Loaded from memory store
</ Response Preferences >

< Calendar Preferences >
{cal_preferences}        # Loaded from memory store
</ Calendar Preferences >
"""
```

### Memory-Aware Prompt Injection

The key insight: **memory content is injected directly into the system prompt** each time the LLM is called:

```python
def llm_call(state: State, store: BaseStore):
    cal_preferences = get_memory(store, ("email_assistant", "cal_preferences"), default_cal_preferences)
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)

    return {
        "messages": [
            llm_with_tools.invoke(
                [{"role": "system", "content": agent_system_prompt_hitl_memory.format(
                    tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
                    background=default_background,
                    response_preferences=response_preferences,   # <-- From store
                    cal_preferences=cal_preferences               # <-- From store
                )}]
                + state["messages"]
            )
        ]
    }
```

---

## 22. Project Structure

```
project_root/
  langgraph.json                    # LangGraph Platform config
  pyproject.toml                    # Package definition
  .env                              # API keys, tokens
  src/
    email_assistant/
      __init__.py
      schemas.py                    # State, RouterSchema, UserPreferences
      prompts.py                    # All system prompts and defaults
      configuration.py              # RunnableConfig helper
      utils.py                      # Email parsing, formatting helpers
      cron.py                       # Cron job graph for scheduled ingestion
      langgraph_101.py              # Basic LangGraph example
      email_assistant.py            # Basic: triage + response
      email_assistant_hitl.py       # + Human-in-the-loop
      email_assistant_hitl_memory.py         # + Memory (BaseStore)
      email_assistant_hitl_memory_gmail.py   # + Real Gmail/Calendar
      tools/
        __init__.py                 # Tool exports
        base.py                     # get_tools(), get_tools_by_name()
        default/
          __init__.py
          email_tools.py            # write_email, Done, Question
          calendar_tools.py         # schedule_meeting, check_calendar_availability
          prompt_templates.py       # Tool description strings
        gmail/
          __init__.py
          gmail_tools.py            # Real Gmail/Calendar API tools
          prompt_templates.py       # Gmail tool description strings
          run_ingest.py             # Email fetching + LangGraph server ingestion
          setup_gmail.py            # OAuth setup script
          setup_cron.py             # Cron job registration
          .secrets/                 # OAuth tokens (gitignored)
      eval/
        __init__.py
        email_dataset.py            # 16 test emails with ground truth
        evaluate_triage.py          # LangSmith evaluation runner
        prompts.py                  # Evaluation judge prompts
  tests/
    conftest.py                     # Pytest config, --agent-module option
    run_all_tests.py                # Test runner with LangSmith integration
    test_response.py                # Tool call + criteria evaluation tests
```

---

## 23. Dependencies

```toml
[project]
requires-python = ">=3.11,<3.14"
dependencies = [
    "langchain>=1.0.0",
    "langchain-core>=1.0.0",
    "langchain-openai>=1.0.0",
    "langgraph>=1.0.0",
    "langsmith[pytest]>=0.4.37",
    "langgraph-cli[inmem]>=0.4.0",       # For `langgraph dev` command
    "langgraph-checkpoint-postgres>=3.0.4", # Production checkpointer
    "psycopg>=3.3.2",                    # PostgreSQL driver
    "google-api-python-client>=2.128.0", # Gmail/Calendar API
    "google-auth-oauthlib",              # OAuth flow
    "python-dotenv",                     # .env loading
    "html2text",                         # HTML email to text
    "pydantic",                          # (via langchain) Schema definitions
]
```

---

## 24. Key Patterns Summary

### Pattern 1: Triage-Then-Act
Classify input first, then route to the appropriate handler. Prevents wasted compute on irrelevant inputs.

### Pattern 2: Subgraph as Node
Compile an inner agent graph and use it as a node in the outer graph. Only the outer graph gets checkpointer/store.

### Pattern 3: Command for Dynamic Routing
Use `Command(goto=, update=)` when a node needs to both update state and choose its destination.

### Pattern 4: Selective HITL
Not all tools need human review. Only interrupt for high-impact actions (send email, schedule meeting). Let low-risk tools (check calendar) execute automatically.

### Pattern 5: Memory via Store + LLM Update
Store preferences in `BaseStore` with namespaces. When humans provide feedback, use an LLM to make **targeted updates** to the stored preferences (never overwrite).

### Pattern 6: Memory Injection into Prompts
Load stored preferences at runtime and inject them into the system prompt. The agent's behavior evolves as memory accumulates.

### Pattern 7: Deterministic Thread IDs
Hash external IDs (like Gmail thread IDs) into UUIDs for consistent thread mapping:
```python
thread_id = str(uuid.UUID(hex=hashlib.md5(raw_id.encode("UTF-8")).hexdigest()))
```

### Pattern 8: State Immutability
Never mutate state objects. Use `message.model_copy(update={...})` to create modified copies.

### Pattern 9: Feedback Loop
Every HITL interaction is a learning opportunity:
- User edits -> update response/cal preferences
- User ignores -> update triage preferences
- User responds with text -> update relevant preferences

### Pattern 10: Progressive Enhancement
Build in layers: basic -> HITL -> memory -> external APIs. Each layer adds capability without breaking the previous one. The graph structure stays the same; nodes get richer.

---

> **Usage:** When building your next ambient agent, use this document as your primary reference. Each section contains the exact code patterns, import paths, and architectural decisions that make a production LangGraph agent work. Copy patterns directly and adapt to your domain.
