"""Defender profile building from on_ball_engagement events."""

import polars as pl


def detect_defensive_actions(events: pl.DataFrame) -> pl.DataFrame:
    """Detect defensive engagements from events."""
    return events.filter(pl.col("event_type") == "on_ball_engagement")


def build_defender_profiles(actions: pl.DataFrame) -> pl.DataFrame:
    """Build defender profiles from on_ball_engagement events."""
    group_cols = ["player_id", "player_name", "team_id"]
    if "team_name" in actions.columns:
        group_cols.append("team_name")

    profiles = actions.group_by(group_cols).agg([
        pl.len().alias("total_engagements"),

        # Defensive outcomes
        pl.col("stop_possession_danger").mean().alias("stop_danger_rate"),
        pl.col("reduce_possession_danger").mean().alias("reduce_danger_rate"),
        pl.col("force_backward").mean().alias("force_backward_rate"),

        # Beaten metrics (lower is better)
        pl.col("beaten_by_possession").mean().alias("beaten_by_possession_rate"),
        pl.col("beaten_by_movement").mean().alias("beaten_by_movement_rate"),

        # Pressing
        pl.col("pressing_chain").mean().alias("pressing_rate"),
        pl.col("pressing_chain_length").mean().alias("avg_pressing_chain_length"),

        # Positioning
        pl.col("interplayer_distance_start").mean().alias("avg_engagement_distance"),
        pl.col("goal_side_start").mean().alias("goal_side_rate"),

        # Zone distribution
        (pl.col("third_start") == "defensive_third").mean().alias("defensive_third_pct"),
        (pl.col("third_start") == "middle_third").mean().alias("middle_third_pct"),
        (pl.col("third_start") == "attacking_third").mean().alias("attacking_third_pct"),
    ])

    # Convert to percentages
    pct_cols = [
        "stop_danger_rate", "reduce_danger_rate", "force_backward_rate",
        "beaten_by_possession_rate", "beaten_by_movement_rate", "pressing_rate",
        "goal_side_rate", "defensive_third_pct", "middle_third_pct", "attacking_third_pct"
    ]
    for col in pct_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns((pl.col(col) * 100).round(1).alias(col))

    # Round numeric columns
    numeric_cols = ["avg_pressing_chain_length", "avg_engagement_distance"]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(pl.col(col).round(2).alias(col))

    return profiles.sort("total_engagements", descending=True)


def filter_defender_profiles(profiles: pl.DataFrame, min_engagements: int = 5) -> pl.DataFrame:
    """Filter to defenders with sufficient engagements."""
    return profiles.filter(pl.col("total_engagements") >= min_engagements)
