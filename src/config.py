"""
Central configuration for the Fantasy Football Co-Pilot data pipeline.

All tunable knobs live here so the rest of the codebase stays free of magic
numbers. Values can be overridden with environment variables where noted.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Project-wide configuration."""

    # ---- Data ----
    # The most recently COMPLETED NFL season. NFL seasons are named by the year
    # they start, so the season played Sept 2024 -> Feb 2025 is "2024".
    stats_season: int = 2024
    # The upcoming season, used for schedule / roster context in draft questions.
    upcoming_season: int = 2025

    # Only keep the fantasy-relevant offensive positions. This keeps the corpus
    # focused and small, which is a deliberate scope-control decision.
    positions: List[str] = field(default_factory=lambda: ["QB", "RB", "WR", "TE"])

    # ---- Embeddings ----
    # Default embedding backend. "local" uses sentence-transformers (free, no API
    # key, runs offline after a one-time model download) so the pipeline is fully
    # reproducible for a grader without any credentials. Set to "openai" to use
    # OpenAI's text-embedding-3-small instead.
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local")
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- Vector store ----
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection_name: str = "nfl_2024_passages"
    top_k: int = 8  # number of passages retrieved per query

    # ---- Generator LLM ----
    # Requires OPENAI_API_KEY to be set in the environment.
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = 0.2
    llm_max_tokens: int = 500

    # ---- Paths ----
    raw_data_dir: str = "data/raw"
    passages_path: str = "data/processed/passages.jsonl"
    outputs_dir: str = "outputs"

    def __post_init__(self) -> None:
        # Make sure the directories we write to exist.
        for path in (self.raw_data_dir, os.path.dirname(self.passages_path),
                     self.chroma_dir, self.outputs_dir):
            os.makedirs(path, exist_ok=True)


# A single shared instance the rest of the code imports.
config = Config()
