"""Data loading utilities for SkillCorner open data.

Note: SkillCorner provides two data types:
- Tracking data (jsonl): Use kloppy.skillcorner.load_open_data() for frame-by-frame positions
- Dynamic events (csv): Load directly for possession-level game intelligence events

This module loads dynamic_events.csv for entry analysis since it contains
pre-computed features like third_start, third_end, lead_to_shot, speed_avg, etc.
"""

import polars as pl
from src.utils.config import GITHUB_BASE_URL, MATCH_IDS, TEAMS


def load_events(match_id: int) -> pl.DataFrame:
    """Load dynamic events for a single match from GitHub."""
    url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_dynamic_events.csv"
    df = pl.read_csv(url, infer_schema_length=10000)
    # Normalize potentially problematic columns to avoid type mismatches when concatenating
    string_cols = ["player_name", "third_start", "third_end", "possession_end_reason",
                   "team_in_possession_phase_type", "inside_defensive_shape_start",
                   "inside_defensive_shape_end"]
    for col in string_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))
    return df.with_columns(pl.lit(match_id).alias("match_id"))


def load_phases(match_id: int) -> pl.DataFrame:
    """Load phases of play for a single match from GitHub."""
    url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_phases_of_play.csv"
    df = pl.read_csv(url)
    return df.with_columns(pl.lit(match_id).alias("match_id"))


def load_all_events(match_ids: list[int] | None = None) -> pl.DataFrame:
    """Load and concatenate events from multiple matches."""
    if match_ids is None:
        match_ids = MATCH_IDS

    dfs = []
    for match_id in match_ids:
        try:
            df = load_events(match_id)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading match {match_id}: {e}")

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal")


def load_all_phases(match_ids: list[int] | None = None) -> pl.DataFrame:
    """Load and concatenate phases from multiple matches."""
    if match_ids is None:
        match_ids = MATCH_IDS

    dfs = []
    for match_id in match_ids:
        try:
            df = load_phases(match_id)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading phases for match {match_id}: {e}")

    if not dfs:
        return pl.DataFrame()

    return pl.concat(dfs, how="diagonal")


def add_team_names(df: pl.DataFrame) -> pl.DataFrame:
    """Add team name column based on team_id."""
    team_mapping = pl.DataFrame({
        "team_id": list(TEAMS.keys()),
        "team_name": list(TEAMS.values()),
    })
    return df.join(team_mapping, on="team_id", how="left")


def get_match_info() -> pl.DataFrame:
    """Get basic info about available matches."""
    url = f"{GITHUB_BASE_URL}/matches.json"
    import json
    import urllib.request

    with urllib.request.urlopen(url) as response:
        matches = json.loads(response.read())

    return pl.DataFrame([
        {
            "match_id": m["id"],
            "home_team": m["home_team"]["short_name"],
            "away_team": m["away_team"]["short_name"],
            "date": m["date_time"][:10],
            "home_score": m.get("home_team_score"),
            "away_score": m.get("away_team_score"),
        }
        for m in matches
    ])
