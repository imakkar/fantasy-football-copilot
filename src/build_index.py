"""
End-to-end index builder.

Runs the full data pipeline with a single command:

    python -m src.build_index

Steps:
  1. Load 2025 weekly stats from nfl_data_py (cached to data/raw/).
  2. Convert each stat row into a natural-language passage.
  3. Save passages to data/processed/passages.jsonl.
  4. Embed all passages and store them in the ChromaDB vector store.
"""

from __future__ import annotations

from .data_loader import load_weekly_stats
from .preprocess import build_passages, save_passages
from .embed_store import build_index


def main() -> None:
    print("=" * 60)
    print("Fantasy Football Co-Pilot :: building data pipeline")
    print("=" * 60)

    weekly = load_weekly_stats()
    print(f"[build_index] loaded {len(weekly)} weekly stat rows")

    passages = build_passages(weekly)
    save_passages(passages)

    build_index(passages)
    print("[build_index] pipeline complete.")


if __name__ == "__main__":
    main()
