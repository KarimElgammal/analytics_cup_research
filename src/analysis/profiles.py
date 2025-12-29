"""Build player profiles from entry-level tracking data.

This module aggregates entry-level data to player profiles for similarity analysis.
"""

import polars as pl
from src.utils.alvarez_profile import MIN_ENTRIES_THRESHOLD, PROFILE_FEATURES


def build_passer_profiles(entries: pl.DataFrame) -> pl.DataFrame:
    """
    Build profiles for players as passers/assisters.

    Aggregates metrics for entries where the player was the passer
    (i.e., the previous event player who enabled the entry).

    Args:
        entries: Classified entries with passer context

    Returns:
        DataFrame with passer-perspective metrics per player
    """
    # Filter to only entries that were assisted
    if "is_assisted" not in entries.columns or "passer_player_id" not in entries.columns:
        return pl.DataFrame()

    assisted_entries = entries.filter(
        pl.col("is_assisted") & pl.col("passer_player_id").is_not_null()
    )

    if len(assisted_entries) == 0:
        return pl.DataFrame()

    # Group by passer
    passer_profiles = assisted_entries.group_by(
        ["passer_player_id", "passer_player_name"]
    ).agg([
        pl.len().alias("total_entry_assists"),
        pl.col("is_dangerous").mean().alias("assist_danger_rate"),
    ])

    return passer_profiles.rename({
        "passer_player_id": "player_id",
        "passer_player_name": "player_name"
    })


def build_player_profiles(entries: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate entry-level data to player profiles.

    Each player gets a profile with averaged metrics across all their
    final third entries.
    """
    # Ensure we have the required columns
    required_cols = ["player_id", "player_name", "team_id"]
    for col in required_cols:
        if col not in entries.columns:
            raise ValueError(f"Missing required column: {col}")

    # Group by player and compute aggregates
    group_cols = ["player_id", "player_name", "team_id"]
    if "team_name" in entries.columns:
        group_cols.append("team_name")

    # Build base aggregation list
    agg_list = [
        # Count
        pl.len().alias("total_entries"),

        # Physical metrics
        pl.col("speed_avg").mean().alias("avg_entry_speed"),
        pl.col("distance_covered").mean().alias("avg_distance"),

        # Spatial metrics
        pl.col("separation_end").mean().alias("avg_separation"),
        pl.col("delta_to_last_defensive_line_end").mean().alias("avg_defensive_line_dist"),
        pl.col("x_end").mean().alias("avg_x_depth"),

        # Tactical metrics
        pl.col("n_passing_options_ahead").mean().alias("avg_passing_options"),
        pl.col("n_teammates_ahead_end").mean().alias("avg_teammates_ahead"),

        # Zone distribution
        (pl.col("entry_zone") == "central").mean().alias("central_pct"),
        (pl.col("entry_zone") == "half_space").mean().alias("half_space_pct"),
        (pl.col("entry_zone") == "wide").mean().alias("wide_pct"),

        # Entry method
        (pl.col("entry_method") == "carry").mean().alias("carry_pct"),

        # Outcomes
        pl.col("is_dangerous").mean().alias("danger_rate"),
        pl.col("is_goal").mean().alias("goal_rate"),

        # Phase distribution
        (pl.col("team_in_possession_phase_type") == "quick_break").mean().alias("quick_break_pct"),
        (pl.col("team_in_possession_phase_type") == "build_up").mean().alias("build_up_pct"),
        (pl.col("team_in_possession_phase_type") == "transition").mean().alias("transition_pct"),

        # SkillCorner advanced metrics (only columns with data)
        pl.col("one_touch").mean().alias("one_touch_pct"),
        pl.col("penalty_area_end").mean().alias("penalty_area_pct"),
        pl.col("n_opponents_bypassed").fill_null(0).mean().alias("avg_opponents_bypassed"),
        pl.col("forward_momentum").mean().alias("forward_momentum_pct"),
    ]

    # Add transition speed metrics if context columns exist
    if "transition_speed_seconds" in entries.columns:
        agg_list.extend([
            pl.col("transition_speed_seconds").mean().alias("avg_transition_speed"),
            pl.col("is_fast_transition").mean().alias("fast_transition_pct"),
        ])

    # Add passer/receiver metrics if context columns exist
    if "is_assisted" in entries.columns:
        agg_list.extend([
            pl.col("is_assisted").mean().alias("assisted_pct"),
            # Danger rate on assisted entries only
            pl.when(pl.col("is_assisted"))
                .then(pl.col("is_dangerous"))
                .otherwise(None)
                .mean().alias("assisted_danger_rate"),
            # Danger rate on unassisted entries only
            pl.when(~pl.col("is_assisted"))
                .then(pl.col("is_dangerous"))
                .otherwise(None)
                .mean().alias("solo_danger_rate"),
        ])

    profiles = entries.group_by(group_cols).agg(agg_list)

    # Build and merge passer profiles if context exists
    if "is_assisted" in entries.columns:
        passer_profiles = build_passer_profiles(entries)
        if len(passer_profiles) > 0:
            profiles = profiles.join(
                passer_profiles,
                on=["player_id", "player_name"],
                how="left"
            )
            # Fill nulls for players with no assists
            profiles = profiles.with_columns([
                pl.col("total_entry_assists").fill_null(0),
                pl.col("assist_danger_rate").fill_null(0.0),
            ])

    # Convert percentages to 0-100 scale for readability
    pct_cols = [
        "central_pct", "half_space_pct", "wide_pct", "carry_pct",
        "danger_rate", "goal_rate", "quick_break_pct", "build_up_pct", "transition_pct",
        "one_touch_pct", "penalty_area_pct", "forward_momentum_pct",
        # New transition and passer/receiver metrics
        "fast_transition_pct", "assisted_pct", "assisted_danger_rate", "solo_danger_rate",
        "assist_danger_rate",
    ]
    for col in pct_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(
                (pl.col(col) * 100).round(1).alias(col)
            )

    # Round other numeric columns
    numeric_cols = [
        "avg_entry_speed", "avg_distance", "avg_separation",
        "avg_defensive_line_dist", "avg_x_depth", "avg_passing_options", "avg_teammates_ahead",
        "avg_opponents_bypassed",
        # New transition metric
        "avg_transition_speed",
    ]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(
                pl.col(col).round(2).alias(col)
            )

    return profiles.sort("total_entries", descending=True)


def filter_profiles(profiles: pl.DataFrame, min_entries: int = MIN_ENTRIES_THRESHOLD) -> pl.DataFrame:
    """
    Filter profiles to players with sufficient entries.

    This helps ensure we're comparing players with enough data for
    reliable profile estimation.
    """
    return profiles.filter(pl.col("total_entries") >= min_entries)


def get_profile_summary(profiles: pl.DataFrame) -> dict:
    """Get summary statistics about the player profiles."""
    return {
        "total_players": len(profiles),
        "avg_entries_per_player": profiles["total_entries"].mean(),
        "max_entries": profiles["total_entries"].max(),
        "min_entries": profiles["total_entries"].min(),
    }


def get_top_players_by_metric(
    profiles: pl.DataFrame,
    metric: str,
    n: int = 10,
    ascending: bool = False
) -> pl.DataFrame:
    """Get top N players ranked by a specific metric."""
    return profiles.sort(metric, descending=not ascending).head(n)


def prepare_features_for_similarity(profiles: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare profile features for similarity computation.

    Returns a DataFrame with just player identifiers and numeric features
    used in similarity scoring.
    """
    # Features used in similarity (matches FEATURE_WEIGHTS keys)
    feature_cols = [
        "avg_separation",
        "central_pct",
        "half_space_pct",
        "danger_rate",
        "avg_entry_speed",
        "avg_passing_options",
        "avg_teammates_ahead",
        "avg_defensive_line_dist",
        "quick_break_pct",
        "avg_distance",
        "goal_rate",
    ]

    # Keep only columns that exist
    available_cols = [c for c in feature_cols if c in profiles.columns]

    id_cols = ["player_id", "player_name", "team_id"]
    if "team_name" in profiles.columns:
        id_cols.append("team_name")

    return profiles.select(id_cols + ["total_entries"] + available_cols)
