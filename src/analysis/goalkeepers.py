"""Goalkeeper profile building from distribution events."""

import polars as pl


def detect_gk_actions(events: pl.DataFrame) -> pl.DataFrame:
    """Detect goalkeeper distribution events."""
    return events.filter(
        (pl.col("player_position") == "GK") &
        (pl.col("event_type") == "player_possession")
    )


def build_goalkeeper_profiles(actions: pl.DataFrame) -> pl.DataFrame:
    """Build goalkeeper profiles from distribution events."""
    group_cols = ["player_id", "player_name", "team_id"]
    if "team_name" in actions.columns:
        group_cols.append("team_name")

    profiles = actions.group_by(group_cols).agg([
        pl.len().alias("total_distributions"),

        # Pass success
        (pl.col("pass_outcome") == "successful").mean().alias("pass_success_rate"),

        # Distribution style
        pl.col("pass_distance").mean().alias("avg_pass_distance"),
        (pl.col("pass_range") == "long").mean().alias("long_pass_pct"),
        (pl.col("pass_range") == "short").mean().alias("short_pass_pct"),
        pl.col("high_pass").mean().alias("high_pass_pct"),
        pl.col("quick_pass").mean().alias("quick_distribution_pct"),
        pl.col("hand_pass").mean().alias("hand_pass_pct"),

        # Passing options & decision making
        pl.col("n_passing_options").mean().alias("avg_passing_options"),

        # Distribution zones (where passes end up)
        (pl.col("third_end") == "middle_third").mean().alias("to_middle_third_pct"),
        (pl.col("third_end") == "attacking_third").mean().alias("to_attacking_third_pct"),

        # Speed of play
        pl.col("speed_avg").mean().alias("avg_speed"),
    ])

    # Convert to percentages
    pct_cols = [
        "pass_success_rate", "long_pass_pct", "short_pass_pct", "high_pass_pct",
        "quick_distribution_pct", "hand_pass_pct", "to_middle_third_pct",
        "to_attacking_third_pct"
    ]
    for col in pct_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns((pl.col(col) * 100).round(1).alias(col))

    # Round numeric columns
    numeric_cols = ["avg_pass_distance", "avg_passing_options", "avg_speed"]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(pl.col(col).round(2).alias(col))

    return profiles.sort("total_distributions", descending=True)


def filter_goalkeeper_profiles(profiles: pl.DataFrame, min_distributions: int = 10) -> pl.DataFrame:
    """Filter to goalkeepers with sufficient distributions."""
    return profiles.filter(pl.col("total_distributions") >= min_distributions)
