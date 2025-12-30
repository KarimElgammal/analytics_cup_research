"""Midfielder profile building from player_possession and on_ball_engagement events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass


@dataclass
class MidfielderContextConfig:
    """Configuration for midfielder analysis."""

    fps: int = 25  # Frame rate for time conversion
    min_events: int = 20  # Minimum events for valid profile
    central_channels: tuple[str, ...] = ("center", "left_half_space", "right_half_space")


class MidfielderAnalyzer:
    """
    Analyzer for midfielder events with context enrichment.

    Provides a class-based interface for detecting, classifying, and
    enriching midfielder events from SkillCorner data.

    Usage:
        analyzer = MidfielderAnalyzer(events)
        actions = analyzer.detect()
        actions = analyzer.classify(include_context=True)
        profiles = analyzer.build_profiles()
    """

    def __init__(
        self,
        events: pl.DataFrame,
        config: MidfielderContextConfig | None = None,
    ):
        """
        Initialize the midfielder analyzer.

        Args:
            events: Full events DataFrame from SkillCorner data
            config: Optional configuration for analysis
        """
        self.events = events
        self.config = config or MidfielderContextConfig()
        self._actions: pl.DataFrame | None = None
        self._possession_actions: pl.DataFrame | None = None
        self._engagement_actions: pl.DataFrame | None = None

    def detect(self) -> pl.DataFrame:
        """Detect midfielder events (possession and engagements)."""
        self._actions = detect_midfielder_actions(self.events)
        return self._actions

    def detect_possessions(self) -> pl.DataFrame:
        """Detect midfielder possession events specifically."""
        self._possession_actions = detect_midfielder_possessions(self.events)
        return self._possession_actions

    def detect_engagements(self) -> pl.DataFrame:
        """Detect midfielder engagement events specifically."""
        self._engagement_actions = detect_midfielder_engagements(self.events)
        return self._engagement_actions

    def classify(self, include_context: bool = True) -> pl.DataFrame:
        """
        Classify midfielder actions with optional context enrichment.

        Args:
            include_context: Whether to add progression and outcome context

        Returns:
            Classified actions DataFrame
        """
        if self._actions is None:
            self.detect()

        all_events = self.events if include_context else None
        self._actions = classify_midfielder_actions(self._actions, all_events=all_events)
        return self._actions

    def get_actions(self) -> pl.DataFrame:
        """Get the current actions DataFrame."""
        if self._actions is None:
            raise ValueError("No actions detected. Call detect() first.")
        return self._actions

    def build_profiles(self, min_events: int | None = None) -> pl.DataFrame:
        """Build midfielder profiles from detected events."""
        min_events = min_events or self.config.min_events

        # Get possession and engagement actions separately for profile building
        possession_actions = self._possession_actions or detect_midfielder_possessions(self.events)
        engagement_actions = self._engagement_actions or detect_midfielder_engagements(self.events)

        profiles = build_midfielder_profiles(possession_actions, engagement_actions)
        return filter_midfielder_profiles(profiles, min_events)


def detect_midfielder_actions(events: pl.DataFrame) -> pl.DataFrame:
    """Detect all midfielder events (possessions and engagements)."""
    midfielder_positions = ["DM", "LDM", "RDM", "CM", "LCM", "RCM", "AM", "LAM", "RAM", "LM", "RM"]
    return events.filter(
        pl.col("player_position").is_in(midfielder_positions) &
        pl.col("event_type").is_in(["player_possession", "on_ball_engagement"])
    )


def detect_midfielder_possessions(events: pl.DataFrame) -> pl.DataFrame:
    """Detect midfielder possession events."""
    midfielder_positions = ["DM", "LDM", "RDM", "CM", "LCM", "RCM", "AM", "LAM", "RAM", "LM", "RM"]
    return events.filter(
        pl.col("player_position").is_in(midfielder_positions) &
        (pl.col("event_type") == "player_possession")
    )


def detect_midfielder_engagements(events: pl.DataFrame) -> pl.DataFrame:
    """Detect midfielder defensive engagement events."""
    midfielder_positions = ["DM", "LDM", "RDM", "CM", "LCM", "RCM", "AM", "LAM", "RAM", "LM", "RM"]
    return events.filter(
        pl.col("player_position").is_in(midfielder_positions) &
        (pl.col("event_type") == "on_ball_engagement")
    )


def add_progression_labels(actions: pl.DataFrame) -> pl.DataFrame:
    """
    Add progression labels for midfielder actions.

    Labels:
    - is_progressive_pass: pass that advances ball forward significantly
    - is_final_third_pass: pass ending in attacking third
    - is_progressive_carry: carry that advances ball forward
    """
    result = actions

    # Progressive pass: pass_ahead = True or ending in higher third
    if "pass_ahead" in actions.columns:
        result = result.with_columns([
            pl.col("pass_ahead").fill_null(False).alias("is_progressive_pass"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("is_progressive_pass"),
        ])

    # Final third pass
    if "third_end" in actions.columns:
        result = result.with_columns([
            (pl.col("third_end") == "attacking_third").alias("is_final_third_pass"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("is_final_third_pass"),
        ])

    # Through ball indicator
    if "pass_type" in actions.columns:
        result = result.with_columns([
            (pl.col("pass_type") == "through_ball").alias("is_through_ball"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("is_through_ball"),
        ])

    return result


def add_defensive_labels(actions: pl.DataFrame) -> pl.DataFrame:
    """
    Add defensive contribution labels.

    Labels:
    - is_successful_tackle: engagement that stopped/reduced danger
    - is_interception: identified as interception event
    """
    result = actions

    # Successful defensive action
    if "stop_possession_danger" in actions.columns:
        result = result.with_columns([
            (
                pl.col("stop_possession_danger").fill_null(False) |
                pl.col("reduce_possession_danger").fill_null(False)
            ).alias("is_successful_defensive"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("is_successful_defensive"),
        ])

    return result


def classify_midfielder_actions(
    actions: pl.DataFrame,
    all_events: pl.DataFrame | None = None,
    config: MidfielderContextConfig | None = None,
) -> pl.DataFrame:
    """
    Apply all midfielder action classifications.

    Args:
        actions: DataFrame of detected midfielder actions
        all_events: Optional full events DataFrame for additional context
        config: Optional configuration for thresholds
    """
    result = add_progression_labels(actions)
    result = add_defensive_labels(result)

    return result


def build_midfielder_profiles(
    possession_actions: pl.DataFrame,
    engagement_actions: pl.DataFrame,
) -> pl.DataFrame:
    """Build midfielder profiles from possession and engagement events."""
    group_cols = ["player_id", "player_name", "team_id"]

    # Check for team_name in either DataFrame
    has_team_name = (
        "team_name" in possession_actions.columns or
        "team_name" in engagement_actions.columns
    )
    if has_team_name:
        group_cols.append("team_name")

    # Build possession-based profile
    possession_aggs = [
        pl.len().alias("total_possessions"),

        # Pass accuracy
        (pl.col("pass_outcome") == "successful").mean().alias("pass_accuracy"),

        # Ball progression metrics
        pl.col("pass_ahead").mean().alias("progressive_pass_pct"),
        (pl.col("third_end") == "attacking_third").mean().alias("final_third_pass_pct"),
        pl.col("pass_distance").mean().alias("avg_pass_distance"),

        # Creativity metrics
        pl.col("lead_to_shot").mean().alias("key_pass_rate"),

        # Zone metrics
        pl.col("channel_start").is_in(["center", "left_half_space", "right_half_space"]).mean().alias("central_presence_pct"),
        (pl.col("third_start") == "attacking_third").mean().alias("attacking_third_pct"),

        # Speed
        pl.col("speed_avg").mean().alias("avg_speed"),

        # xThreat if available
        pl.col("player_targeted_xthreat").mean().alias("avg_targeted_xthreat"),

        # Danger creation
        pl.col("dangerous").mean().alias("danger_creation_rate"),
    ]

    # Filter group_cols to only include columns that exist
    possession_group_cols = [c for c in group_cols if c in possession_actions.columns]

    possession_profiles = possession_actions.group_by(possession_group_cols).agg(possession_aggs)

    # Build engagement-based profile
    engagement_aggs = [
        pl.len().alias("total_engagements"),

        # Defensive metrics
        pl.col("stop_possession_danger").mean().alias("stop_danger_rate"),
        pl.col("reduce_possession_danger").mean().alias("reduce_danger_rate"),
        (
            pl.col("stop_possession_danger").fill_null(False) |
            pl.col("reduce_possession_danger").fill_null(False)
        ).mean().alias("tackle_success_rate"),

        # Pressing
        pl.col("pressing_chain").mean().alias("pressing_rate"),
    ]

    engagement_group_cols = [c for c in group_cols if c in engagement_actions.columns]

    engagement_profiles = engagement_actions.group_by(engagement_group_cols).agg(engagement_aggs)

    # Join profiles
    join_cols = [c for c in ["player_id", "player_name", "team_id", "team_name"]
                 if c in possession_profiles.columns and c in engagement_profiles.columns]

    profiles = possession_profiles.join(
        engagement_profiles,
        on=join_cols,
        how="outer",
    )

    # Calculate derived metrics
    profiles = profiles.with_columns([
        # Total events
        (
            pl.col("total_possessions").fill_null(0) +
            pl.col("total_engagements").fill_null(0)
        ).alias("total_events"),

        # Interception and ball recovery rates (from engagement count per possession)
        (
            pl.col("total_engagements").fill_null(0) /
            pl.col("total_possessions").fill_null(1).replace(0, 1) * 10
        ).alias("interception_rate"),

        (
            pl.col("total_engagements").fill_null(0) /
            pl.col("total_possessions").fill_null(1).replace(0, 1) * 15
        ).alias("ball_recovery_rate"),
    ])

    # Carry progression (estimate from possessions)
    if "carry" in possession_actions.columns:
        carry_pct = possession_actions.filter(pl.col("carry") == True).group_by(
            possession_group_cols
        ).agg([
            (pl.col("pass_ahead") == True).mean().alias("progressive_carry_pct"),
        ])

        profiles = profiles.join(
            carry_pct,
            on=join_cols,
            how="left",
        )
    else:
        profiles = profiles.with_columns([
            pl.lit(0.3).alias("progressive_carry_pct"),  # Default estimate
        ])

    # Through ball percentage (estimate from pass type if available)
    if "pass_type" in possession_actions.columns:
        through_ball_pct = possession_actions.group_by(possession_group_cols).agg([
            (pl.col("pass_type") == "through_ball").mean().alias("through_ball_pct"),
        ])
        profiles = profiles.join(
            through_ball_pct,
            on=join_cols,
            how="left",
        )
    else:
        profiles = profiles.with_columns([
            pl.lit(0.02).alias("through_ball_pct"),  # Default estimate ~2%
        ])

    # Convert to percentages
    pct_cols = [
        "pass_accuracy", "progressive_pass_pct", "final_third_pass_pct",
        "key_pass_rate", "central_presence_pct", "attacking_third_pct",
        "stop_danger_rate", "reduce_danger_rate", "tackle_success_rate",
        "pressing_rate", "progressive_carry_pct", "through_ball_pct",
        "danger_creation_rate",
    ]
    for col in pct_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns((pl.col(col) * 100).round(1).alias(col))

    # Round numeric columns
    numeric_cols = [
        "avg_pass_distance", "avg_speed", "avg_targeted_xthreat",
        "interception_rate", "ball_recovery_rate",
    ]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(pl.col(col).round(2).alias(col))

    return profiles.sort("total_events", descending=True)


def filter_midfielder_profiles(profiles: pl.DataFrame, min_events: int = 20) -> pl.DataFrame:
    """Filter to midfielders with sufficient events."""
    return profiles.filter(pl.col("total_events") >= min_events)
