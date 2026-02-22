"""
report_graph.py — Compiled report writing subgraph.

Graph flow:
  START
    → report_supervisor   (plans sections, fans out via Send())
    → [writer_agent × N]  (parallel section writers, each returns report_sections update)
    → section_reviewer    (reviews sections; re-dispatches rejects or proceeds)
    → report_assembler    (combines sections → report_draft)
    → END

Compiled WITHOUT checkpointer or store — inherited from outer graph at runtime.

Exported as `report_graph` for import by main_graph.py.
"""

from langgraph.graph import END, START, StateGraph

from strategic_analyst.schemas import AgentState

from .supervisor import report_assembler, report_supervisor, section_reviewer
from .writer_agent import writer_agent

# ── Build graph ───────────────────────────────────────────────────────────────

_builder = StateGraph(AgentState)

_builder.add_node("report_supervisor", report_supervisor)
_builder.add_node("writer_agent", writer_agent)
_builder.add_node("section_reviewer", section_reviewer)
_builder.add_node("report_assembler", report_assembler)

# Static edges
_builder.add_edge(START, "report_supervisor")
# report_supervisor uses Command(goto=[Send(...)]) — no static edge needed to writer_agent

# After all parallel writer_agents finish, section_reviewer runs
_builder.add_edge("writer_agent", "section_reviewer")
# section_reviewer uses Command to route:
#   - Command(goto=[Send("writer_agent", ...)]) for re-dispatch
#   - Command(goto="report_assembler") when all approved

_builder.add_edge("report_assembler", END)

# ── Compile (no checkpointer / store — inherited from outer graph) ────────────
report_graph = _builder.compile()
