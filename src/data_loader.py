"""
Data loading for the Fantasy Football Co-Pilot.

Pulls structured NFL data from the open-source `nfl_data_py` package (which mirrors
the maintained `nflverse` project). A single package supplies weekly player
statistics, rosters, depth charts, schedules, and injuries, so the pipeline needs
no web scraping.

Every loader returns a pandas DataFrame and caches a raw CSV copy under
`data/raw/` so re-runs are fast and fully reproducible.
"""

from __future__ import annotations

import os
import pandas as pd

from .config import config


def _cache_path(name: str) -> str:
    return os.path.join(config.raw_data_dir, f"{name}.csv")


def _load_or_cache(name: str, fetch_fn) -> pd.DataFrame:
    """Return a cached CSV if present, otherwise fetch, cache, and return."""
    path = _cache_path(name)
    if os.path.exists(path):
        print(f"[data_loader] loading cached {name} from {path}")
        return pd.read_csv(path, low_memory=False)

    print(f"[data_loader] fetching {name} from nfl_data_py ...")
    df = fetch_fn()
    df.to_csv(path, index=False)
    print(f"[data_loader] cached {name} -> {path} ({len(df)} rows)")
    return df


def load_weekly_stats(season: int | None = None) -> pd.DataFrame:
    """Weekly player statistics for the given season (defaults to config season).

    Includes fantasy points (standard and PPR), which we use both as passage
    content and as ground truth in evaluation.
    """
    import nfl_data_py as nfl

    season = season or config.stats_season
    df = _load_or_cache(
        f"weekly_{season}",
        lambda: nfl.import_weekly_data([season]),
    )
    # Keep only fantasy-relevant offensive positions.
    if "position" in df.columns:
        df = df[df["position"].isin(config.positions)].reset_index(drop=True)
    return df


def load_rosters(season: int | None = None) -> pd.DataFrame:
    """Team rosters for the given season."""
    import nfl_data_py as nfl

    season = season or config.stats_season
    return _load_or_cache(
        f"rosters_{season}",
        lambda: nfl.import_seasonal_rosters([season]),
    )


def load_schedule(season: int | None = None) -> pd.DataFrame:
    """Game schedule for the given season (defaults to the upcoming season)."""
    import nfl_data_py as nfl

    season = season or config.upcoming_season
    return _load_or_cache(
        f"schedule_{season}",
        lambda: nfl.import_schedules([season]),
    )


def load_injuries(season: int | None = None) -> pd.DataFrame:
    """Injury reports for the given season. Returns empty DataFrame if unavailable."""
    import nfl_data_py as nfl

    season = season or config.stats_season
    try:
        return _load_or_cache(
            f"injuries_{season}",
            lambda: nfl.import_injuries([season]),
        )
    except Exception as exc:  # some seasons/pace may lack injury feeds
        print(f"[data_loader] injuries unavailable for {season}: {exc}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Smoke test: load everything and print shapes.
    weekly = load_weekly_stats()
    print("weekly stats:", weekly.shape)
    print(weekly.head())
