"""
report_saver.py — Final node that persists an approved report to Supabase.

Actions:
  1. Extract ReportMetadata via LLM structured output (utility model)
  2. Insert report record into `reports` table
  3. Chunk + embed report content → insert into `report_chunks` table
     (same BGE-M3 / 1024-dim embedding as the main `chunks` table)
  4. Update episodic_memory with session summary
  5. Return a confirmation AI message
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from supabase import create_client

from strategic_analyst.configuration import AgentConfiguration
from strategic_analyst.memory import (
    NAMESPACE_KEYS,
    episodic_memory_ns,
    update_memory_with_llm_async,
)
from strategic_analyst.prompts import REPORT_METADATA_SYSTEM_PROMPT
from strategic_analyst.schemas import AgentState, ReportMetadata
from strategic_analyst.tools.rag_tool import _embed_query


async def save_report_node(
    state: AgentState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Persist the approved report to Supabase and update episodic memory.
    """
    cfg = AgentConfiguration.from_runnable_config(config)
    user_id = cfg.user_id
    session_id = state.get("session_id", str(uuid.uuid4()))
    final_report = state.get("final_report") or state.get("report_draft", "")
    report_format = state.get("report_format", "markdown")

    if not final_report:
        return {
            "messages": [{"role": "assistant", "content": "No report content to save."}],
            "current_phase": "end",
        }

    # ── Step 1: Extract metadata ───────────────────────────────────────────────
    llm = ChatOpenAI(
        model=cfg.utility_model_name,
        api_key=os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN"),
        base_url=os.getenv("OVH_API_BASE_URL", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
        temperature=0.0,
    ).with_structured_output(ReportMetadata)

    metadata: ReportMetadata = await llm.ainvoke([
        {"role": "system", "content": REPORT_METADATA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract metadata from this report:\n\n{final_report[:4000]}",
        },
    ])

    # ── Step 2: Insert into `reports` table ───────────────────────────────────
    report_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    report_record = {
        "id": report_id,
        "user_id": user_id,
        "session_id": session_id,
        "title": metadata.title,
        "content": final_report,
        "format": report_format,
        "topic_tags": metadata.topic_tags,
        "project_name": metadata.project_name,
        "executive_summary": metadata.executive_summary,
        "frameworks_used": metadata.frameworks_used,
        "metadata": {},
        "created_at": now_iso,
    }

    loop = asyncio.get_event_loop()
    supabase = await loop.run_in_executor(
        None,
        lambda: create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY"),
        ),
    )

    await loop.run_in_executor(
        None,
        lambda: supabase.table("reports").insert(report_record).execute(),
    )

    # ── Step 3: Chunk + embed → `report_chunks` ───────────────────────────────
    chunks = _chunk_text(final_report, chunk_size=500)

    async def _embed_and_insert(index: int, chunk_text: str) -> None:
        embedding = await _embed_query(chunk_text)
        chunk_record = {
            "id": str(uuid.uuid4()),
            "report_id": report_id,
            "user_id": user_id,
            "content": chunk_text,
            "embedding": embedding,
            "chunk_index": index,
            "heading": None,
        }
        await loop.run_in_executor(
            None,
            lambda: supabase.table("report_chunks").insert(chunk_record).execute(),
        )

    await asyncio.gather(*[_embed_and_insert(i, c) for i, c in enumerate(chunks)])

    # ── Step 4: Update episodic memory ────────────────────────────────────────
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await update_memory_with_llm_async(
        store=store,
        namespace=episodic_memory_ns(user_id),
        key=NAMESPACE_KEYS["episodic_memory"],
        context_messages=[
            {
                "role": "user",
                "content": (
                    f"On {today}, completed a research session. "
                    f"Report title: '{metadata.title}'. "
                    f"Summary: {metadata.executive_summary} "
                    f"Report ID: {report_id}."
                ),
            }
        ],
        update_reason=f"Session completed on {today}. Add report summary to episodic memory.",
        model_name=cfg.utility_model_name,
    )

    # ── Step 5: Return confirmation ───────────────────────────────────────────
    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    f"Report saved successfully.\n\n"
                    f"**Title:** {metadata.title}\n"
                    f"**Report ID:** `{report_id}`\n"
                    f"**Format:** {report_format}\n"
                    f"**Tags:** {', '.join(metadata.topic_tags)}\n\n"
                    f"The report has been added to the company knowledge base "
                    f"and will be retrievable in future research sessions."
                ),
            }
        ],
        "current_phase": "end",
    }


def _chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
