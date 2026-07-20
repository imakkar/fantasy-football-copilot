"""
Preprocessing: turn structured NFL stat rows into short natural-language passages.

RAG retrieval works over text, so each weekly player-stat row is converted into a
compact English sentence that an embedding model can represent well. One passage
per player per week keeps each chunk focused and self-contained.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from .config import config


def _safe(value, default=0):
    """Return a clean numeric/string value, coercing NaN to a default."""
    if pd.isna(value):
        return default
    return value


def weekly_row_to_passage(row: pd.Series) -> Dict:
    """Convert one weekly-stats row into a passage dict {id, text, metadata}."""
    name = _safe(row.get("player_display_name") or row.get("player_name"), "Unknown")
    pos = _safe(row.get("position"), "NA")
    team = _safe(row.get("recent_team") or row.get("team"), "NA")
    week = int(_safe(row.get("week"), 0))
    season = int(_safe(row.get("season"), config.stats_season))
    opp = _safe(row.get("opponent_team"), "NA")

    ppr = float(_safe(row.get("fantasy_points_ppr"), 0.0))
    std = float(_safe(row.get("fantasy_points"), 0.0))

    # Position-relevant stat lines.
    pass_yds = int(_safe(row.get("passing_yards"), 0))
    pass_td = int(_safe(row.get("passing_tds"), 0))
    ints = int(_safe(row.get("interceptions"), 0))
    rush_yds = int(_safe(row.get("rushing_yards"), 0))
    rush_td = int(_safe(row.get("rushing_tds"), 0))
    rec = int(_safe(row.get("receptions"), 0))
    targets = int(_safe(row.get("targets"), 0))
    rec_yds = int(_safe(row.get("receiving_yards"), 0))
    rec_td = int(_safe(row.get("receiving_tds"), 0))

    text = (
        f"In Week {week} of the {season} NFL season, {name} ({pos}, {team}) "
        f"played against {opp} and scored {ppr:.1f} PPR fantasy points "
        f"({std:.1f} standard). "
    )
    if pos == "QB":
        text += (
            f"He threw for {pass_yds} yards and {pass_td} touchdowns with {ints} "
            f"interceptions, and added {rush_yds} rushing yards and {rush_td} "
            f"rushing touchdowns."
        )
    elif pos == "RB":
        text += (
            f"He rushed for {rush_yds} yards and {rush_td} touchdowns, and caught "
            f"{rec} of {targets} targets for {rec_yds} yards and {rec_td} receiving "
            f"touchdowns."
        )
    else:  # WR / TE
        text += (
            f"He caught {rec} of {targets} targets for {rec_yds} receiving yards "
            f"and {rec_td} touchdowns."
        )

    passage_id = f"{season}_wk{week}_{str(name).replace(' ', '_')}_{team}"
    metadata = {
        "player": str(name),
        "position": str(pos),
        "team": str(team),
        "week": week,
        "season": season,
        "opponent": str(opp),
        "fantasy_points_ppr": ppr,
        "fantasy_points": std,
    }
    return {"id": passage_id, "text": text, "metadata": metadata}


def build_passages(weekly: pd.DataFrame) -> List[Dict]:
    """Convert every weekly-stats row into a passage."""
    passages = [weekly_row_to_passage(row) for _, row in weekly.iterrows()]
    print(f"[preprocess] built {len(passages)} passages")
    return passages


def save_passages(passages: List[Dict], path: str | None = None) -> str:
    """Write passages to a JSONL file for inspection and reuse."""
    path = path or config.passages_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"[preprocess] saved passages -> {path}")
    return path


def load_passages(path: str | None = None) -> List[Dict]:
    """Read passages back from JSONL."""
    path = path or config.passages_path
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    from .data_loader import load_weekly_stats

    weekly = load_weekly_stats()
    passages = build_passages(weekly)
    save_passages(passages)
    print("Example passage:\n", passages[0]["text"])
