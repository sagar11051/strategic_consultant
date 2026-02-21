# Placeholder — wired fully in Step 05 / Step 13 (Main Graph)
# Minimal compilable graph so `langgraph dev` can import this module.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from strategic_analyst.schemas import AgentState

_builder = StateGraph(AgentState)
_builder.add_node("placeholder", lambda state: state)
_builder.add_edge(START, "placeholder")
_builder.add_edge("placeholder", END)

graph = _builder.compile(checkpointer=MemorySaver())
