"""
main_graph.py — Top-level compiled graph for the Strategic Analyst Agent.

Graph flow:
  START
    → context_loader        (parallel RAG + memory load)
    → greeting_node         (personalised greeting)
    → planner_agent         (research plan; Command → hitl_plan_gate)
    → hitl_plan_gate        (HITL 1; Command → research_subgraph | planner_agent | END)
    → research_subgraph     (compiled research subgraph)
    → hitl_discovery_gate   (HITL 2; Command → report_subgraph | research_subgraph | END)
    → report_subgraph       (compiled report-writing subgraph)
    → hitl_final_gate       (HITL 3; Command → save_report_node | report_subgraph | ...)
    → save_report_node      (persist to Supabase)
    → END

Module-level `graph` is compiled WITHOUT checkpointer/store so that
`langgraph dev` can load it — the platform injects its own persistence.

For testing, use build_graph(store=InMemoryStore(), checkpointer=MemorySaver())
to pass explicit instances.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

from strategic_analyst.nodes.context_loader import context_loader, greeting_node
from strategic_analyst.nodes.hitl_gates import (
    hitl_discovery_gate,
    hitl_final_gate,
    hitl_plan_gate,
)
from strategic_analyst.nodes.planner import planner_agent
from strategic_analyst.nodes.report_saver import save_report_node
from strategic_analyst.schemas import AgentInput, AgentState
from strategic_analyst.subgraphs.research.research_graph import research_graph
from strategic_analyst.subgraphs.report.report_graph import report_graph

load_dotenv()


def _build_builder() -> StateGraph:
    """Shared graph topology — nodes and edges only, no checkpointer/store."""
    builder = StateGraph(AgentState, input_schema=AgentInput)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("context_loader", context_loader)
    builder.add_node("greeting_node", greeting_node)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("hitl_plan_gate", hitl_plan_gate)
    builder.add_node("research_subgraph", research_graph)   # compiled subgraph
    builder.add_node("hitl_discovery_gate", hitl_discovery_gate)
    builder.add_node("report_subgraph", report_graph)       # compiled subgraph
    builder.add_node("hitl_final_gate", hitl_final_gate)
    builder.add_node("save_report_node", save_report_node)

    # ── Static edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "context_loader")
    builder.add_edge("context_loader", "greeting_node")
    builder.add_edge("greeting_node", "planner_agent")
    # planner_agent → hitl_plan_gate via Command
    # hitl_plan_gate → research_subgraph | planner_agent | END via Command
    builder.add_edge("research_subgraph", "hitl_discovery_gate")
    # hitl_discovery_gate → report_subgraph | research_subgraph | END via Command
    builder.add_edge("report_subgraph", "hitl_final_gate")
    # hitl_final_gate → save_report_node | report_subgraph | ... via Command
    builder.add_edge("save_report_node", END)

    return builder


def build_graph(store: BaseStore | None = None, checkpointer=None):
    """
    Build and compile the graph with explicit checkpointer + store.

    Use this in tests:
        g = build_graph(store=InMemoryStore(), checkpointer=MemorySaver())

    Args:
        store:        BaseStore instance (required for store-aware nodes).
        checkpointer: LangGraph checkpointer (required for interrupt/resume).

    Returns:
        CompiledStateGraph with the provided persistence backends.
    """
    if store is None:
        store = InMemoryStore()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return _build_builder().compile(checkpointer=checkpointer, store=store)


# ── Module-level graph (what langgraph.json points to) ───────────────────────
# Compiled WITHOUT checkpointer/store — langgraph dev injects its own.
graph = _build_builder().compile()
