"""
task_agent.py — Individual research worker node.

Receives a TaskAgentState dict via Send() from the research supervisor.
Runs a ReAct-style tool loop (up to MAX_ITERATIONS) until:
  - The agent calls the Done tool (signals completion), OR
  - Max iterations are exhausted

After the loop, performs a structured extraction call to produce a
ResearchFinding object. Returns {"research_findings": {task_id: finding_dict}}
which is merged into AgentState.research_findings via the merge_dicts reducer.
"""

from __future__ import annotations

import os

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.prompts import TASK_AGENT_SYSTEM_PROMPT
from strategic_analyst.schemas import ResearchFinding
from strategic_analyst.tools.base import TASK_AGENT_TOOLS, get_tools

_MAX_ITERATIONS = 8
_OVH_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


def _ovh_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", _OVH_BASE),
        **kwargs,
    )


async def task_agent(state: dict, store: BaseStore, config: RunnableConfig) -> dict:
    """
    Research task worker node. Called via Send() with a TaskAgentState dict.

    Runs a ReAct loop using TASK_AGENT_TOOLS (semantic_search, hybrid_search,
    web_search_tool, Done). After the loop, extracts a structured ResearchFinding.

    Returns:
        {"research_findings": {task_id: <finding dict>}}
    """
    cfg = AgentConfiguration.from_runnable_config(config)

    task_id = state.get("task_id", "unknown_task")
    question = state.get("question", "")
    data_sources = state.get("data_sources", ["company_db", "web"])
    context = state.get("context", "")
    supervisor_critique = state.get("supervisor_critique", "")

    # Build tools
    tools = get_tools(TASK_AGENT_TOOLS)
    tools_by_name = {t.name: t for t in tools}

    retry_info = (
        f"\n\n**SUPERVISOR CRITIQUE — you MUST address this in your research:**\n{supervisor_critique}"
        if supervisor_critique
        else ""
    )

    system_prompt = TASK_AGENT_SYSTEM_PROMPT.format(
        task_id=task_id,
        question=question,
        data_sources=", ".join(data_sources),
        context=context or "No prior context provided.",
        retry_info=retry_info,
    )

    # ReAct LLM — tool_choice="required" forces a tool call every turn
    react_llm = _ovh_llm(
        model=cfg.model_name,
        temperature=0.1,
    ).bind_tools(tools, tool_choice="required")

    messages: list = [{"role": "system", "content": system_prompt}]
    done_summary = ""

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for _ in range(_MAX_ITERATIONS):
        response = await react_llm.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        tool_results = []
        hit_done = False

        for tc in response.tool_calls:
            tool_name = tc["name"]
            args = tc["args"]
            call_id = tc["id"]

            if tool_name == "Done":
                done_summary = args.get("summary", "")
                tool_results.append({
                    "role": "tool",
                    "content": "Task marked as complete.",
                    "tool_call_id": call_id,
                })
                hit_done = True
                break

            tool_obj = tools_by_name.get(tool_name)
            if tool_obj is not None:
                try:
                    result = await tool_obj.ainvoke(args)
                    content = str(result)
                except Exception as exc:
                    content = f"Tool error ({tool_name}): {exc}"
            else:
                content = f"Unknown tool: {tool_name}"

            tool_results.append({
                "role": "tool",
                "content": content,
                "tool_call_id": call_id,
            })

        messages.extend(tool_results)

        if hit_done:
            break

    # ── Structured extraction ─────────────────────────────────────────────────
    transcript_parts: list[str] = []
    for m in messages[1:]:  # skip system prompt
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "tool":
                transcript_parts.append(f"[Tool result]: {content[:800]}")
        elif hasattr(m, "content") and m.content:
            transcript_parts.append(f"[Assistant]: {m.content[:800]}")
        elif getattr(m, "tool_calls", None):
            names = [tc["name"] for tc in m.tool_calls]
            transcript_parts.append(f"[Tool calls]: {', '.join(names)}")

    if done_summary:
        transcript_parts.append(f"[Done summary]: {done_summary}")

    transcript = "\n".join(transcript_parts) if transcript_parts else "No research was conducted."

    extraction_llm = _ovh_llm(
        model=cfg.model_name,
        temperature=0.0,
    ).with_structured_output(ResearchFinding)

    try:
        finding: ResearchFinding = await extraction_llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "You are a research analyst extracting a structured finding from a completed research session.\n"
                    "Extract the ResearchFinding from the conversation below.\n"
                    f"task_id must be exactly: {task_id}\n"
                    "Be specific — use actual data points found. Note missing info in gaps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\n\n"
                    f"Research conversation:\n{transcript}"
                ),
            },
        ])
        finding_dict = finding.model_dump()
        finding_dict["task_id"] = task_id  # enforce correct task_id
    except Exception as exc:
        finding_dict = {
            "task_id": task_id,
            "answer": done_summary or "Research completed — see transcript for details.",
            "evidence": [],
            "sources": [],
            "confidence": "low",
            "gaps": f"Structured extraction failed: {exc}",
        }

    return {"research_findings": {task_id: finding_dict}}
