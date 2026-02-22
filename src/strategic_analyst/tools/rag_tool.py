"""
RAG tools for the strategic knowledge base.

Two search methods are exposed as separate LangChain tools so the agent LLM
can choose which to use and issue multiple queries per turn:

  semantic_search  — pure BGE-M3 vector similarity (cosine distance via pgvector HNSW)
  hybrid_search    — RRF fusion of vector similarity + BM25 full-text search (DEFAULT)

Both call Supabase RPC functions with the same names (see CLAUDE.md Database Schema §3).

Embedding:
  - Model : BAAI/bge-m3  (1024 dims)
  - Client: OVH AI Endpoints HTTP API  (no local model — no download required)
  - URL   : OVH_EMBEDDING_ENDPOINT_URL  (defaults to OVH BGE-M3 endpoint)
  - Auth  : Bearer OVH_KEY  (same token used for LLM calls)

Supabase calls:
  - Sync supabase-py client called via asyncio.run_in_executor for async safety.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache

import aiohttp
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from supabase import create_client, Client


# ── OVH Embedding API ─────────────────────────────────────────────────────────

_OVH_EMBEDDING_URL_DEFAULT = "https://bge-m3.endpoints.kepler.ai.cloud.ovh.net/api/text2vec"


def _get_ovh_token() -> str:
    """Return OVH auth token. OVH_KEY is the primary; falls back to OVH_AI_ENDPOINTS_ACCESS_TOKEN."""
    token = os.getenv("OVH_KEY") or os.getenv("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
    if not token:
        raise EnvironmentError(
            "OVH auth token not found. Set OVH_KEY in your .env file."
        )
    return token


async def _embed_query(text: str) -> list[float]:
    """
    Call the OVH BGE-M3 embedding endpoint and return a 1024-dim dense vector.

    Request  : POST OVH_EMBEDDING_ENDPOINT_URL
               Authorization: Bearer <OVH_KEY>
               Body: {"inputs": "<text>"}

    Response : [[float, ...]]  (HuggingFace inference API format — list of embeddings)
               The outer list is the batch dimension; we always send one item.
    """
    url   = os.getenv("OVH_EMBEDDING_ENDPOINT_URL", _OVH_EMBEDDING_URL_DEFAULT)
    token = _get_ovh_token()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"inputs": text},
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    # Normalise response shape:
    #   [[float, ...]]  → result[0]   (batch of 1, HuggingFace format)
    #   [float, ...]    → result       (bare vector)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return result


# ── Supabase client (cached sync, called via executor) ────────────────────────

@lru_cache(maxsize=1)
def _get_supabase() -> Client:
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


async def _call_rpc(func_name: str, params: dict) -> list[dict]:
    """Execute a Supabase RPC call in a thread executor (non-blocking)."""
    client = _get_supabase()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.rpc(func_name, params).execute(),
    )
    return response.data or []


# ── Result formatter ───────────────────────────────────────────────────────────

def _format_results(rows: list[dict], score_label: str = "Relevance") -> str:
    if not rows:
        return "No relevant documents found in the knowledge base."
    parts: list[str] = []
    for i, row in enumerate(rows, 1):
        heading = f" — {row['heading']}" if row.get("heading") else ""
        page    = f" (p.{row['page_number']})" if row.get("page_number") else ""
        score   = row.get("score", 0.0)
        parts.append(
            f"[{i}] {row['document_title']}{heading}{page} | {score_label}: {score:.4f}\n"
            f"{row['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Tool 1: Semantic Search ────────────────────────────────────────────────────

class SemanticSearchInput(BaseModel):
    query_text: str = Field(description="Natural language question or topic to search for")
    limit: int = Field(default=10, description="Number of results to return (1–20)")


@tool(args_schema=SemanticSearchInput)
async def semantic_search(query_text: str, limit: int = 10) -> str:
    """
    Search the strategic knowledge base using vector similarity (BGE-M3, cosine distance).

    Best for:
    - Broad conceptual questions ("what is our competitive position?")
    - Paraphrased queries where the exact words don't appear in source documents
    - Finding thematically related content across different phrasings

    Prefer hybrid_search for specific company names, financial figures, metrics,
    or any query where exact keyword matches matter alongside semantic meaning.

    You can call this tool multiple times with different phrasings of the same question
    to improve retrieval coverage. Each call is independent.

    Always cite document_title and page_number in your final response.
    """
    embedding = await _embed_query(query_text)
    rows = await _call_rpc(
        "semantic_search",
        {"query_embedding": embedding, "match_count": min(limit, 20)},
    )
    return _format_results(rows, score_label="Similarity")


# ── Tool 2: Hybrid Search (DEFAULT) ───────────────────────────────────────────

class HybridSearchInput(BaseModel):
    query_text: str = Field(description="Natural language question or search terms")
    limit: int = Field(default=10, description="Number of results to return (1–20)")


@tool(args_schema=HybridSearchInput)
async def hybrid_search(query_text: str, limit: int = 10) -> str:
    """
    Search the strategic knowledge base using vector similarity AND BM25 keyword
    matching, fused via Reciprocal Rank Fusion (RRF).

    DEFAULT SEARCH — use this for most queries, especially:
    - Specific company names, product names, brand terms
    - Financial figures, percentages, dates, and metrics
    - Technical or industry-specific terminology
    - Any query where exact keyword matches matter alongside semantic meaning

    You can call this tool multiple times with different phrasings, or break a
    complex question into focused sub-queries for broader coverage.

    Always cite document_title and page_number in your final response.
    """
    embedding = await _embed_query(query_text)
    rows = await _call_rpc(
        "hybrid_search",
        {
            "query_embedding": embedding,
            "query_text":      query_text,
            "match_count":     min(limit, 20),
        },
    )
    return _format_results(rows, score_label="RRF Score")
