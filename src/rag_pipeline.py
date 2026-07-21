"""
RAG pipeline: retrieve relevant passages, then ask the LLM to answer using only
that retrieved context.

This is the primary system under test for RQ1 (does retrieval grounding improve
faithfulness and outcome accuracy over a no-retrieval baseline?).
"""

from __future__ import annotations

from typing import Dict, List

from .config import config
from .embed_store import retrieve
from .entities import extract_filters, build_where_clause
from .generator import generate

SYSTEM_PROMPT = (
    "You are Fantasy Football Co-Pilot, an assistant that helps managers make "
    "draft and start/sit decisions. You are given retrieved statistics from the "
    "completed 2024 NFL season. Answer the user's question using ONLY the provided "
    "context. Cite the specific stats you used. If the context does not contain "
    "enough information, say so plainly rather than guessing. Be concise and give a "
    "clear recommendation."
)


def _format_context(hits: List[Dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] {hit['text']}")
    return "\n".join(lines)


def answer(query: str, top_k: int | None = None) -> Dict:
    """Run the full RAG pipeline for a single query.

    Returns a dict with the generated answer plus the retrieved context, so the
    transparency sidebar (and the evaluation harness) can inspect what was used.
    """
    # Extract players/week from the question and filter retrieval to matching
    # passages. This keeps the retrieved context on-topic (right players, right
    # week) instead of relying on semantic similarity alone. If the filter matches
    # nothing (e.g. a player had a bye that week), fall back to plain retrieval.
    filters = extract_filters(query)
    where = build_where_clause(filters["players"], filters["week"])
    hits = retrieve(query, top_k=top_k, where=where)
    if not hits and where is not None:
        hits = retrieve(query, top_k=top_k)

    context = _format_context(hits)

    user_prompt = (
        f"Question: {query}\n\n"
        f"Retrieved 2024-season context:\n{context}\n\n"
        f"Answer using only the context above and cite the stats you rely on."
    )
    answer_text = generate(SYSTEM_PROMPT, user_prompt)
    return {
        "query": query,
        "answer": answer_text,
        "retrieved": hits,
        "mode": "rag",
    }


if __name__ == "__main__":
    out = answer("In 2024, who was the better start in week 5: a WR1 on Detroit or a WR1 on Minnesota?")
    print(out["answer"])
