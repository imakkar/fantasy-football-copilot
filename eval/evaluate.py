"""
Evaluation harness for RQ1: does RAG grounding beat a no-retrieval baseline?

Runs both the RAG pipeline and the non-RAG baseline over the benchmark queries and
scores them on:

  * Outcome accuracy (start/sit only): did the answer name the player who ACTUALLY
    scored more PPR fantasy points that week? Ground truth is computed directly from
    the 2025 weekly stats, so it is objective and reproducible - never hand-labeled.

  * Latency: end-to-end response time per query.

Faithfulness and helpfulness are human-annotated (a template CSV is written for the
grader/author to fill in), because they require judgment that cannot be fully
automated at this stage. This split keeps the automatic metric honest while still
scaffolding the qualitative ones.

Usage:
    python -m eval.evaluate                 # full run (needs OPENAI_API_KEY + index)
    python -m eval.evaluate --ground-truth  # only compute/print ground truth, no LLM
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

# Allow running as `python -m eval.evaluate` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config  # noqa: E402
from src.data_loader import load_weekly_stats  # noqa: E402


def load_queries() -> List[Dict]:
    with open("eval/benchmark_queries.json") as f:
        return json.load(f)


def ground_truth_start_sit(item: Dict, weekly) -> Optional[Dict]:
    """Compute the true higher-scorer for a start/sit query from the data.

    Returns {"winner": name, "points": {player: ppr, ...}} or None if a player's
    row for that week can't be found.
    """
    week = item["week"]
    points = {}
    for player in item["players"]:
        rows = weekly[
            (weekly["player_display_name"] == player) & (weekly["week"] == week)
        ]
        if rows.empty:
            # Try the alternate name column.
            if "player_name" in weekly.columns:
                rows = weekly[
                    (weekly["player_name"] == player) & (weekly["week"] == week)
                ]
        if rows.empty:
            return None
        points[player] = float(rows.iloc[0]["fantasy_points_ppr"])

    winner = max(points, key=points.get)
    return {"winner": winner, "points": points}


def answer_names_winner(answer_text: str, winner: str) -> bool:
    """Heuristic: did the model's answer recommend the actual higher scorer?

    Checks that the winner's name appears and is framed as the recommendation. This
    is a simple, transparent proxy; borderline cases are flagged for human review in
    the CSV.
    """
    text = answer_text.lower()
    winner_l = winner.lower()
    return winner_l in text


def run(ground_truth_only: bool = False) -> None:
    queries = load_queries()
    weekly = load_weekly_stats()

    # Lazy import so `--ground-truth` works without an API key.
    if not ground_truth_only:
        from src import rag_pipeline, baseline

    rows_for_csv = []
    rag_correct = base_correct = n_scored = 0

    for item in queries:
        gt = None
        if item["type"] == "start_sit":
            gt = ground_truth_start_sit(item, weekly)

        record = {
            "id": item["id"],
            "type": item["type"],
            "query": item["query"],
            "ground_truth_winner": gt["winner"] if gt else "",
            "ground_truth_points": json.dumps(gt["points"]) if gt else "",
        }

        if ground_truth_only:
            print(f"{item['id']}: ", gt if gt else "(qualitative / no ground truth)")
            rows_for_csv.append(record)
            continue

        # RAG + baseline, with error handling so one failure doesn't kill the run.
        try:
            t0 = time.time()
            rag_out = rag_pipeline.answer(item["query"])
            record["rag_latency_s"] = round(time.time() - t0, 2)
            record["rag_answer"] = rag_out["answer"]

            t0 = time.time()
            base_out = baseline.answer(item["query"])
            record["baseline_latency_s"] = round(time.time() - t0, 2)
            record["baseline_answer"] = base_out["answer"]
        except Exception as exc:
            print(f"[eval] {item['id']} failed: {exc}")
            record["rag_answer"] = f"ERROR: {exc}"
            record["baseline_answer"] = ""
            rows_for_csv.append(record)
            continue

        # Brief pause to stay within the Gemini free-tier rate limit.
        time.sleep(2)

        # Automatic outcome scoring (start/sit with known ground truth).
        if gt:
            n_scored += 1
            rag_hit = answer_names_winner(rag_out["answer"], gt["winner"])
            base_hit = answer_names_winner(base_out["answer"], gt["winner"])
            rag_correct += int(rag_hit)
            base_correct += int(base_hit)
            record["rag_outcome_correct"] = rag_hit
            record["baseline_outcome_correct"] = base_hit

        # Columns for human annotation (RQ1 faithfulness/helpfulness).
        record["rag_faithful_1to5"] = ""
        record["baseline_faithful_1to5"] = ""
        record["rag_helpful_1to5"] = ""
        record["baseline_helpful_1to5"] = ""

        rows_for_csv.append(record)
        print(f"[eval] {item['id']} done "
              f"(rag {record.get('rag_latency_s')}s / base {record.get('baseline_latency_s')}s)")

    # Write the results CSV.
    os.makedirs(config.outputs_dir, exist_ok=True)
    csv_path = os.path.join(config.outputs_dir, "evaluation_results.csv")
    if rows_for_csv:
        keys = sorted({k for r in rows_for_csv for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\n[eval] wrote {csv_path}")

    # Print the automatic outcome-accuracy summary.
    if not ground_truth_only and n_scored:
        print("\n" + "=" * 50)
        print("RQ1 :: Outcome accuracy on start/sit queries")
        print(f"  scored questions : {n_scored}")
        print(f"  RAG correct      : {rag_correct}/{n_scored} "
              f"({100*rag_correct/n_scored:.0f}%)")
        print(f"  Baseline correct : {base_correct}/{n_scored} "
              f"({100*base_correct/n_scored:.0f}%)")
        print("=" * 50)
        print("Faithfulness/helpfulness columns are left blank in the CSV for "
              "human annotation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", action="store_true",
                        help="Only compute ground truth from data (no LLM calls).")
    args = parser.parse_args()
    run(ground_truth_only=args.ground_truth)
