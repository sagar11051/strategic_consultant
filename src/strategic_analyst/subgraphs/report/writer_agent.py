"""
writer_agent.py — Report section writer worker node.

Receives a WriterAgentState dict via Send() from the report supervisor.
Uses LONG_CONTEXT_MODEL (gpt-oss-20b) for high-quality prose generation.
Runs a ReAct loop with RAG and memory tools to gather supplementary context,
then writes the assigned section.

Returns {"report_sections": {section_id: {title, content, instructions}}}
which is merged into AgentState.report_sections via the merge_dicts reducer.
"""

from __future__ import annotations

import os

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from strategic_analyst.prompts import REPORT_WRITER_SYSTEM_PROMPT
from strategic_analyst.tools.base import REPORT_WRITER_TOOLS, get_tools, make_tool_node

_MAX_ITERATIONS = 6
_OVH_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
# Hardcoded: writer uses the long-context model for quality prose
_WRITER_MODEL = "gpt-oss-20b"


def _ovh_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", _OVH_BASE),
        **kwargs,
    )


async def writer_agent(state: dict, store: BaseStore, config: RunnableConfig) -> dict:
    """
    Report section writer node. Called via Send() with a WriterAgentState dict.

    Runs a ReAct loop using REPORT_WRITER_TOOLS (semantic_search, hybrid_search,
    memory_search_tool, Done) with the LONG_CONTEXT_MODEL for richer writing.

    Returns:
        {"report_sections": {section_id: {title, content, instructions}}}
    """
    section_id = state.get("section_id", "unknown_section")
    section_title = state.get("section_title", "")
    section_instructions = state.get("section_instructions", "")
    research_findings = state.get("research_findings", {})
    supervisor_summary = state.get("supervisor_summary", "")
    user_preferences = state.get("user_preferences", "")
    user_profile = state.get("user_profile", "")
    supervisor_critique = state.get("supervisor_critique", "")
    user_id = state.get("user_id", "")

    # Build tools (memory_search_tool needs store + user_id)
    tools = get_tools(REPORT_WRITER_TOOLS)
    tools_by_name = {t.name: t for t in tools}

    # make_tool_node for memory tool dispatch
    _tool_executor = make_tool_node(tools, store, user_id)

    critique_section = (
        f"\n\n**REVISION REQUIRED — address this critique before writing:**\n{supervisor_critique}"
        if supervisor_critique
        else ""
    )

    system_prompt = REPORT_WRITER_SYSTEM_PROMPT.format(
        section_title=section_title,
        section_instructions=section_instructions,
        user_profile=user_profile,
        user_preferences=user_preferences,
        supervisor_summary=supervisor_summary,
        critique_section=critique_section,
    )

    # Condense research findings for context
    findings_text = "\n\n".join(
        f"**{tid}:** {data.get('answer', '')}"
        for tid, data in research_findings.items()
    ) or "No research findings available."

    # Writer LLM — long-context model, tool_choice="auto" so it can write prose
    writer_llm = _ovh_llm(
        model=_WRITER_MODEL,
        temperature=0.3,
    ).bind_tools(tools, tool_choice="auto")

    messages: list = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Research findings for reference:\n\n{findings_text}\n\n"
                "Now write the section. You may use search tools for additional context "
                "before writing. When the section is complete, call Done."
            ),
        },
    ]

    section_content = ""
    done_summary = ""

    # ── ReAct loop ────────────────────────────────────────────────────────────
    for _ in range(_MAX_ITERATIONS):
        response = await writer_llm.ainvoke(messages)
        messages.append(response)

        # Accumulate any prose content the LLM wrote
        if hasattr(response, "content") and response.content:
            section_content = response.content  # keep the latest / longest response

        if not getattr(response, "tool_calls", None):
            # No tool calls — LLM finished writing
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
                    "content": "Section marked as complete.",
                    "tool_call_id": call_id,
                })
                hit_done = True
                break

            # Use make_tool_node for memory tools; direct ainvoke for others
            if tool_name in ("memory_search_tool", "write_memory_tool"):
                # Wrap in a minimal state dict to reuse make_tool_node
                fake_state = {"messages": [response]}
                result_dict = await _tool_executor(fake_state)
                for msg in result_dict.get("messages", []):
                    tool_results.append(msg)
                break  # make_tool_node handles all tool_calls in the message
            else:
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

        if tool_results:
            messages.extend(tool_results)

        if hit_done:
            break

    # ── Extract final section content ─────────────────────────────────────────
    # Use the last substantial assistant message as the section content
    if not section_content:
        for m in reversed(messages):
            if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None):
                section_content = m.content
                break

    if not section_content:
        section_content = done_summary or f"Section '{section_title}' could not be generated."

    section_data = {
        "title": section_title,
        "content": section_content,
        "instructions": section_instructions,
    }

    return {"report_sections": {section_id: section_data}}
