"""
Lightweight entity extraction for retrieval filtering.

Pulls the player names and week number out of a free-text question so the retriever
can filter the vector store to exactly the relevant passages before ranking. This
fixes the precision problem where pure semantic search returns a player's stats from
the wrong week (or only one of the two players being compared).

Deliberately simple and dependency-free: player names are matched against the known
set from the corpus, and the week is found with a regex. No LLM call is needed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

from .preprocess import load_passages


@lru_cache(maxsize=1)
def known_players() -> frozenset:
    """Return the set of player names present in the corpus (cached)."""
    names = {p["metadata"]["player"] for p in load_passages()}
    return frozenset(names)


def extract_week(query: str) -> Optional[int]:
    """Find a 'Week N' reference in the query, if present."""
    match = re.search(r"week\s+(\d+)", query, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_players(query: str) -> List[str]:
    """Return known player names that appear in the query (case-insensitive)."""
    q = query.lower()
    found = [name for name in known_players() if name.lower() in q]
    # Longest-first avoids partial-name collisions (e.g. "Brown" inside two names).
    found.sort(key=len, reverse=True)
    return found


def extract_filters(query: str) -> Dict:
    """Return {'players': [...], 'week': int|None} extracted from the query."""
    return {"players": extract_players(query), "week": extract_week(query)}


def build_where_clause(players: List[str], week: Optional[int]) -> Optional[Dict]:
    """Translate extracted entities into a ChromaDB `where` metadata filter.

    - Players present -> restrict to those players.
    - Week present    -> restrict to that week.
    Returns None when there is nothing to filter on (falls back to plain search).
    """
    conditions = []
    if players:
        conditions.append({"player": {"$in": players}})
    if week is not None:
        conditions.append({"week": week})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
