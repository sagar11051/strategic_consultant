"""
research_graph.py — Compiled research subgraph.

Graph flow:
  START
    → research_supervisor   (reads research_tasks, fans out via Send())
    → [task_agent × N]      (parallel ReAct workers, each returns research_findings update)
    → supervisor_review     (reviews findings; re-dispatches rejects or proceeds)
    → discovery_synthesiser (synthesises findings → supervisor_summary)
    → END

Compiled WITHOUT checkpointer or store — these are inherited from the outer
main graph at runtime (per CLAUDE.md architecture rule §1).

Exported as `research_graph` for import by main_graph.py.
"""

from langgraph.graph import END, START, StateGraph

from strategic_analyst.schemas import AgentState

from .supervisor import (
    discovery_synthesiser,
    research_supervisor,
    supervisor_review,
)
from .task_agent import task_agent

# ── Build graph ───────────────────────────────────────────────────────────────

_builder = StateGraph(AgentState)

_builder.add_node("research_supervisor", research_supervisor)
_builder.add_node("task_agent", task_agent)
_builder.add_node("supervisor_review", supervisor_review)
_builder.add_node("discovery_synthesiser", discovery_synthesiser)

# Static edges
_builder.add_edge(START, "research_supervisor")
# research_supervisor uses Command(goto=[Send(...)]) — no static edge needed to task_agent

# After all parallel task_agents finish, supervisor_review runs
_builder.add_edge("task_agent", "supervisor_review")
# supervisor_review uses Command to route:
#   - Command(goto=[Send("task_agent", ...)]) for re-dispatch
#   - Command(goto="discovery_synthesiser") when all approved

_builder.add_edge("discovery_synthesiser", END)

# ── Compile (no checkpointer / store — inherited from outer graph) ────────────
research_graph = _builder.compile()
