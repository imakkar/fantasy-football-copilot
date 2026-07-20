"""
Non-RAG baseline: the same LLM answering the same question with NO retrieved
context.

This is the control condition for RQ1. Comparing this against the RAG pipeline
isolates the effect of retrieval grounding on faithfulness and outcome accuracy.
"""

from __future__ import annotations

from typing import Dict

from .config import config

SYSTEM_PROMPT = (
    "You are Fantasy Football Co-Pilot, an assistant that helps managers make "
    "draft and start/sit decisions. Answer the user's question about the 2025 NFL "
    "season from your own knowledge. Be concise and give a clear recommendation."
)


def answer(query: str) -> Dict:
    """Answer a query with no retrieval (parametric knowledge only)."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return {
        "query": query,
        "answer": resp.choices[0].message.content,
        "retrieved": [],
        "mode": "baseline",
    }


if __name__ == "__main__":
    out = answer("In 2025, who was the better start in week 5: a WR1 on Detroit or a WR1 on Minnesota?")
    print(out["answer"])
