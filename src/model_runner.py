"""
Single-command entry point that demonstrates the end-to-end system.

    python -m src.model_runner

Loads the benchmark queries, runs both the RAG pipeline and the non-RAG baseline
on the first few, and saves the generated answers (with retrieved context) to the
outputs/ directory. This is the Milestone 4 "runnable pipeline" entry point; in
Milestone 3 it doubles as a smoke test that the whole stack is wired together.

Requires:
  * A built index (run `python -m src.build_index` first).
  * OPENAI_API_KEY set in the environment (for the generator LLM).
"""

from __future__ import annotations

import json
import os

from .config import config
from . import rag_pipeline, baseline


def _load_benchmark() -> list:
    with open("eval/benchmark_queries.json") as f:
        return json.load(f)


def main(num_samples: int = 5) -> None:
    os.makedirs(config.outputs_dir, exist_ok=True)
    queries = _load_benchmark()[:num_samples]

    results = []
    for i, item in enumerate(queries, 1):
        q = item["query"]
        print(f"\n[{i}/{len(queries)}] {q}")

        rag_out = rag_pipeline.answer(q)
        base_out = baseline.answer(q)

        print("  RAG      :", rag_out["answer"][:120].replace("\n", " "), "...")
        print("  BASELINE :", base_out["answer"][:120].replace("\n", " "), "...")

        results.append(
            {"id": item["id"], "query": q, "rag": rag_out, "baseline": base_out}
        )

    out_path = os.path.join(config.outputs_dir, "sample_generations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also write a human-readable text file of the samples.
    txt_path = os.path.join(config.outputs_dir, "samples.txt")
    with open(txt_path, "w") as f:
        for r in results:
            f.write(f"QUERY: {r['query']}\n\n")
            f.write(f"--- RAG ANSWER ---\n{r['rag']['answer']}\n\n")
            f.write(f"--- BASELINE ANSWER ---\n{r['baseline']['answer']}\n\n")
            f.write("=" * 70 + "\n\n")

    print(f"\nSaved {len(results)} sample generations -> {out_path}")
    print(f"Saved human-readable samples -> {txt_path}")


if __name__ == "__main__":
    main()
