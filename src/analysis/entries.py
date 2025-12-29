"""Final third entry detection and classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from typing import Callable


@dataclass
class EntryContextConfig:
    """Configuration for entry context analysis."""

    fps: int = 25  # Frame rate for time conversion
    fast_transition_threshold_seconds: float = 3.0  # Threshold for "fast" transitions

    @property
    def fast_transition_threshold_frames(self) -> int:
        """Convert seconds threshold to frames."""
        return int(self.fast_transition_threshold_seconds * self.fps)


class EntryAnalyzer:
    """
    Analyzer for final third entries with context enrichment.

    Provides a class-based interface for detecting, classifying, and
    enriching final third entries with transition speed and passer context.

    Usage:
        analyzer = EntryAnalyzer(events)
        entries = analyzer.detect()
        entries = analyzer.classify(include_context=True)
        profiles = analyzer.build_profiles()
    """

    def __init__(
        self,
        events: pl.DataFrame,
        config: EntryContextConfig | None = None,
    ):
        """
        Initialize the entry analyzer.

        Args:
            events: Full events DataFrame from SkillCorner data
            config: Optional configuration for context analysis
        """
        self.events = events
        self.config = config or EntryContextConfig()
        self._entries: pl.DataFrame | None = None

    def detect(self) -> pl.DataFrame:
        """Detect final third entries from events."""
        self._entries = detect_entries(self.events)
        return self._entries

    def classify(self, include_context: bool = True) -> pl.DataFrame:
        """
        Classify entries with optional context enrichment.

        Args:
            include_context: Whether to add transition and passer context

        Returns:
            Classified entries DataFrame
        """
        if self._entries is None:
            self.detect()

        all_events = self.events if include_context else None
        self._entries = classify_entries(self._entries, all_events=all_events)
        return self._entries

    def get_entries(self) -> pl.DataFrame:
        """Get the current entries DataFrame."""
        if self._entries is None:
            raise ValueError("No entries detected. Call detect() first.")
        return self._entries

    def get_summary(self) -> dict:
        """Get summary statistics about entries."""
        return get_entry_summary(self.get_entries())


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


def add_transition_context(entries: pl.DataFrame, all_events: pl.DataFrame) -> pl.DataFrame:
    """
    Add transition speed context to entries.

    For each entry, calculates:
    - frames_from_phase_start: frames elapsed from possession phase start to entry
    - transition_speed_seconds: above converted to seconds (assuming 25fps)
    - is_fast_transition: boolean indicating if entry happened within 3 seconds (75 frames)

    Args:
        entries: DataFrame of detected final third entries
        all_events: Full events DataFrame with all possession events

    Returns:
        entries DataFrame with added transition context columns
    """
    # Use phase_index (more reliable than player_possession_phase_index)
    if "phase_index" not in all_events.columns:
        return entries

    # Get phase start frames: minimum frame_start per phase per match
    phase_starts = all_events.group_by(
        ["match_id", "phase_index"]
    ).agg([
        pl.col("frame_start").min().alias("phase_frame_start")
    ])

    # Join phase start info to entries
    entries_with_context = entries.join(
        phase_starts,
        on=["match_id", "phase_index"],
        how="left"
    )

    # Calculate transition metrics
    return entries_with_context.with_columns([
        # Frames from phase start to entry
        (pl.col("frame_start") - pl.col("phase_frame_start")).alias("frames_from_phase_start"),
        # Convert to seconds (25 fps)
        ((pl.col("frame_start") - pl.col("phase_frame_start")) / 25.0).alias("transition_speed_seconds"),
        # Fast transition = within 3 seconds (75 frames)
        ((pl.col("frame_start") - pl.col("phase_frame_start")) <= 75).alias("is_fast_transition"),
    ]).drop("phase_frame_start")


def add_passer_context(entries: pl.DataFrame, all_events: pl.DataFrame) -> pl.DataFrame:
    """
    Add passer context to entries.

    For each entry, identifies the previous event in the same possession phase
    and extracts the player who enabled the entry (the "passer").

    Args:
        entries: DataFrame of detected final third entries
        all_events: Full events DataFrame

    Returns:
        entries DataFrame with passer_player_id, passer_player_name, is_assisted columns
    """
    # Use phase_index for grouping
    if "phase_index" not in all_events.columns:
        return entries

    # Get all events sorted by frame within each match/phase
    events_sorted = all_events.sort(["match_id", "phase_index", "frame_start"])

    # Add row numbers within each phase (same team only)
    events_with_seq = events_sorted.with_columns([
        pl.col("frame_start").rank("ordinal").over(
            ["match_id", "team_id", "phase_index"]
        ).alias("event_seq")
    ])

    # Create lookup for previous event in sequence
    prev_events = events_with_seq.select([
        "match_id", "team_id", "phase_index",
        (pl.col("event_seq") + 1).alias("event_seq"),  # Shift forward to get "next" event's previous
        pl.col("player_id").alias("passer_player_id"),
        pl.col("player_name").alias("passer_player_name"),
    ])

    # Add sequence numbers to entries
    entries_with_seq = entries.join(
        events_with_seq.select([
            "match_id", "team_id", "phase_index", "frame_start", "event_seq"
        ]),
        on=["match_id", "team_id", "phase_index", "frame_start"],
        how="left"
    )

    # Join to get passer info
    entries_with_passer = entries_with_seq.join(
        prev_events,
        on=["match_id", "team_id", "phase_index", "event_seq"],
        how="left"
    )

    # Mark entries where passer is different from receiver (actual assist scenario)
    return entries_with_passer.with_columns([
        (
            pl.col("passer_player_id").is_not_null() &
            (pl.col("passer_player_id") != pl.col("player_id"))
        ).alias("is_assisted"),
    ]).drop("event_seq")


def add_entry_context(entries: pl.DataFrame, all_events: pl.DataFrame) -> pl.DataFrame:
    """
    Add comprehensive context to entries including:
    - Transition speed metrics
    - Passer/assister information

    Args:
        entries: DataFrame of detected final third entries
        all_events: Full events DataFrame

    Returns:
        entries DataFrame with added context columns
    """
    result = entries
    result = add_transition_context(result, all_events)
    result = add_passer_context(result, all_events)
    return result


def classify_entries(entries: pl.DataFrame, all_events: pl.DataFrame | None = None) -> pl.DataFrame:
    """Apply all entry classifications.

    Args:
        entries: DataFrame of detected entries
        all_events: Optional full events DataFrame for context (transition speed, passer info)
    """
    result = (
        entries
        .pipe(classify_entry_zone)
        .pipe(classify_entry_method)
        .pipe(classify_entry_side)
        .pipe(add_danger_labels)
    )

    # Add entry context if full events provided
    if all_events is not None:
        result = add_entry_context(result, all_events)

    return result


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
