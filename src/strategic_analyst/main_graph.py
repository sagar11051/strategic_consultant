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

This is the ONLY graph compiled with checkpointer + store.
Subgraphs are imported pre-compiled and inherit both at runtime.

Exported `graph` at module level — what langgraph.json points to.
"""

from __future__ import annotations

import os
import uuid

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


def build_graph(store: BaseStore | None = None, checkpointer=None):
    """
    Build and compile the main graph.

    Args:
        store:        BaseStore instance. Defaults to InMemoryStore().
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver().

    Returns:
        CompiledStateGraph ready for invocation.
    """
    if store is None:
        store = InMemoryStore()
    if checkpointer is None:
        checkpointer = MemorySaver()

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
    # Context init → greeting → planning
    builder.add_edge(START, "context_loader")
    builder.add_edge("context_loader", "greeting_node")
    builder.add_edge("greeting_node", "planner_agent")

    # planner_agent uses Command(goto="hitl_plan_gate") — no static edge needed

    # hitl_plan_gate uses Command → research_subgraph | planner_agent | END

    # After research finishes → discovery review
    builder.add_edge("research_subgraph", "hitl_discovery_gate")

    # hitl_discovery_gate uses Command → report_subgraph | research_subgraph | END

    # After report draft finishes → final review
    builder.add_edge("report_subgraph", "hitl_final_gate")

    # hitl_final_gate uses Command →
    #   save_report_node | report_subgraph | research_subgraph |
    #   planner_agent | hitl_final_gate | END

    # After save → done
    builder.add_edge("save_report_node", END)

    # ── Compile with checkpointer + store ─────────────────────────────────────
    return builder.compile(checkpointer=checkpointer, store=store)


# ── Module-level graph (what langgraph.json points to) ───────────────────────
graph = build_graph()
