"""
Single-command entry point for the end-to-end system.

    python src/model_runner.py            # run on the default 5 sample queries
    python src/model_runner.py --num 10   # run on 10

Loads the preprocessed corpus (via the vector store), loads the generative model
(Gemini by default), runs both the RAG pipeline and the no-retrieval baseline on a
handful of benchmark queries, and saves the generated outputs to `outputs/`.

Requirements:
  * A built index. Run `python src/build_index.py` first if `data/chroma/` is empty.
  * GEMINI_API_KEY set in the environment (free key from https://aistudio.google.com),
    or LLM_BACKEND=openai with OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a plain script (`python src/model_runner.py`) by putting the repo
# root on the path so the `src` and `utils` packages import correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src import rag_pipeline, baseline
from src.generator import backend_name
from utils.helpers import Timer, get_logger

log = get_logger("model_runner")

BENCHMARK_PATH = "eval/benchmark_queries.json"


def _load_benchmark(num_samples: int) -> list:
    if not os.path.exists(BENCHMARK_PATH):
        raise FileNotFoundError(
            f"Benchmark file not found at {BENCHMARK_PATH}. Run from the repo root."
        )
    with open(BENCHMARK_PATH) as f:
        return json.load(f)[:num_samples]


def _index_ready() -> bool:
    """Best-effort check that the vector store has been built."""
    return os.path.isdir(config.chroma_dir) and bool(os.listdir(config.chroma_dir))


def main(num_samples: int = 5) -> None:
    os.makedirs(config.outputs_dir, exist_ok=True)

    if not _index_ready():
        log.warning(
            "Vector store at '%s' looks empty. Run `python src/build_index.py` first.",
            config.chroma_dir,
        )

    log.info("Generator backend: %s", backend_name())
    queries = _load_benchmark(num_samples)

    results = []
    for i, item in enumerate(queries, 1):
        q = item["query"]
        log.info("[%d/%d] %s", i, len(queries), q)
        try:
            with Timer() as t_rag:
                rag_out = rag_pipeline.answer(q)
            with Timer() as t_base:
                base_out = baseline.answer(q)
        except Exception as exc:  # keep going even if one query fails
            log.error("query %s failed: %s", item["id"], exc)
            continue

        results.append({
            "id": item["id"],
            "query": q,
            "rag_answer": rag_out["answer"],
            "rag_latency_s": t_rag.seconds,
            "num_retrieved": len(rag_out["retrieved"]),
            "baseline_answer": base_out["answer"],
            "baseline_latency_s": t_base.seconds,
        })
        log.info("    rag %.2fs (%d passages) | baseline %.2fs",
                 t_rag.seconds, len(rag_out["retrieved"]), t_base.seconds)

    if not results:
        log.error("No outputs were generated. Check your API key and index.")
        sys.exit(1)

    # Machine-readable output.
    json_path = os.path.join(config.outputs_dir, "sample_generations.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable output.
    txt_path = os.path.join(config.outputs_dir, "samples.txt")
    with open(txt_path, "w") as f:
        f.write("Fantasy Football Co-Pilot -- sample generations\n")
        f.write(f"Generator backend: {backend_name()}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"QUERY [{r['id']}]: {r['query']}\n\n")
            f.write(f"--- RAG ANSWER ({r['rag_latency_s']}s, "
                    f"{r['num_retrieved']} passages retrieved) ---\n{r['rag_answer']}\n\n")
            f.write(f"--- BASELINE ANSWER ({r['baseline_latency_s']}s, no retrieval) ---\n"
                    f"{r['baseline_answer']}\n\n")
            f.write("=" * 70 + "\n\n")

    log.info("Saved %d samples -> %s and %s", len(results), json_path, txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Co-Pilot on sample queries.")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of benchmark queries to run (default 5).")
    args = parser.parse_args()
    main(num_samples=args.num)
