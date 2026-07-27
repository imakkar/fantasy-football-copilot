"""
Central configuration for the Fantasy Football Co-Pilot data pipeline.

All tunable knobs live here so the rest of the codebase stays free of magic numbers.

Precedence (highest first):
  1. Environment variables (where noted).
  2. configs/model_config.yaml, if present.
  3. The dataclass defaults below.
"""

import os
from dataclasses import dataclass, field
from typing import List

_CONFIG_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "model_config.yaml",
)


def _load_yaml() -> dict:
    """Load configs/model_config.yaml if PyYAML and the file are available.

    Returns an empty dict on any failure so the pipeline still runs on defaults.
    """
    try:
        import yaml  # optional dependency
        with open(_CONFIG_YAML) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_YAML = _load_yaml()


def _cfg(section: str, key: str, default):
    """Fetch a value from the loaded YAML, falling back to a default."""
    return _YAML.get(section, {}).get(key, default)


@dataclass
class Config:
    """Project-wide configuration."""

    # ---- Data ----
    # The most recently COMPLETED NFL season. NFL seasons are named by the year
    # they start, so the season played Sept 2024 -> Feb 2025 is "2024".
    stats_season: int = _cfg("data", "stats_season", 2024)
    # The upcoming season, used for schedule / roster context in draft questions.
    upcoming_season: int = _cfg("data", "upcoming_season", 2025)

    # Only keep the fantasy-relevant offensive positions. This keeps the corpus
    # focused and small, which is a deliberate scope-control decision.
    positions: List[str] = field(
        default_factory=lambda: _cfg("data", "positions", ["QB", "RB", "WR", "TE"])
    )

    # ---- Embeddings ----
    # Default embedding backend. "local" uses sentence-transformers (free, no API
    # key, runs offline after a one-time model download) so the pipeline is fully
    # reproducible for a grader without any credentials. Set to "openai" to use
    # OpenAI's text-embedding-3-small instead.
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", _cfg("embeddings", "backend", "local"))
    local_embedding_model: str = _cfg("embeddings", "local_model", "sentence-transformers/all-MiniLM-L6-v2")
    openai_embedding_model: str = _cfg("embeddings", "openai_model", "text-embedding-3-small")

    # ---- Vector store ----
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    collection_name: str = _cfg("vector_store", "collection_name", "nfl_2024_passages")
    top_k: int = _cfg("vector_store", "top_k", 8)  # passages retrieved per query

    # ---- Generator LLM ----
    # Backend selects the provider. "gemini" uses Google's free AI Studio tier
    # (needs GEMINI_API_KEY, no billing); "openai" uses OpenAI (needs OPENAI_API_KEY).
    llm_backend: str = os.getenv("LLM_BACKEND", _cfg("generator", "backend", "gemini"))
    gemini_model: str = os.getenv("GEMINI_MODEL", _cfg("generator", "gemini_model", "gemini-2.5-flash"))
    openai_model: str = os.getenv("OPENAI_MODEL", _cfg("generator", "openai_model", "gpt-4o-mini"))
    llm_temperature: float = _cfg("generator", "temperature", 0.2)
    llm_max_tokens: int = _cfg("generator", "max_tokens", 500)

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
