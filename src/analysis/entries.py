"""Final third entry detection and classification."""

import polars as pl


def detect_entries(events: pl.DataFrame) -> pl.DataFrame:
    """
    Detect final third entries from events.

    Entry = player_possession event where:
    - third_start is NOT attacking_third
    - third_end IS attacking_third
    """
    return events.filter(
        (pl.col("event_type") == "player_possession") &
        (pl.col("third_start") != "attacking_third") &
        (pl.col("third_end") == "attacking_third")
    )


def classify_entry_zone(entries: pl.DataFrame) -> pl.DataFrame:
    """Add entry zone classification based on channel_end."""
    return entries.with_columns(
        pl.when(pl.col("channel_end") == "center")
        .then(pl.lit("central"))
        .when(pl.col("channel_end").str.contains("half_space"))
        .then(pl.lit("half_space"))
        .otherwise(pl.lit("wide"))
        .alias("entry_zone")
    )


def classify_entry_method(entries: pl.DataFrame) -> pl.DataFrame:
    """Add entry method classification (carry vs pass reception)."""
    return entries.with_columns(
        pl.when(pl.col("carry") == True)
        .then(pl.lit("carry"))
        .otherwise(pl.lit("pass_reception"))
        .alias("entry_method")
    )


def classify_entry_side(entries: pl.DataFrame) -> pl.DataFrame:
    """Add entry side (left/right/central)."""
    return entries.with_columns(
        pl.when(pl.col("channel_end").str.contains("left"))
        .then(pl.lit("left"))
        .when(pl.col("channel_end").str.contains("right"))
        .then(pl.lit("right"))
        .otherwise(pl.lit("central"))
        .alias("entry_side")
    )


def add_danger_labels(entries: pl.DataFrame) -> pl.DataFrame:
    """Add binary danger labels based on lead_to_shot and lead_to_goal."""
    return entries.with_columns([
        pl.col("lead_to_shot").alias("is_dangerous"),
        pl.col("lead_to_goal").alias("is_goal"),
    ])


def classify_entries(entries: pl.DataFrame) -> pl.DataFrame:
    """Apply all entry classifications."""
    return (
        entries
        .pipe(classify_entry_zone)
        .pipe(classify_entry_method)
        .pipe(classify_entry_side)
        .pipe(add_danger_labels)
    )


def get_entry_summary(entries: pl.DataFrame) -> dict:
    """Get summary statistics about entries."""
    total = len(entries)
    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "by_zone": entries.group_by("entry_zone").len().to_dicts(),
        "by_method": entries.group_by("entry_method").len().to_dicts(),
        "by_phase": entries.group_by("team_in_possession_phase_type").len().to_dicts(),
        "shots": entries.filter(pl.col("lead_to_shot")).height,
        "goals": entries.filter(pl.col("lead_to_goal")).height,
    }
