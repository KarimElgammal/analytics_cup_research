"""Goalkeeper profile building from distribution events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass


@dataclass
class GoalkeeperContextConfig:
    """Configuration for goalkeeper distribution context analysis."""

    fps: int = 25  # Frame rate for time conversion
    quick_distribution_threshold_seconds: float = 3.0  # Threshold for "quick" distributions

    @property
    def quick_distribution_threshold_frames(self) -> int:
        """Convert seconds threshold to frames."""
        return int(self.quick_distribution_threshold_seconds * self.fps)


class GoalkeeperAnalyzer:
    """
    Analyzer for goalkeeper distributions with context enrichment.

    Provides a class-based interface for detecting, classifying, and
    enriching goalkeeper distributions with transition speed and outcome context.

    Usage:
        analyzer = GoalkeeperAnalyzer(events)
        actions = analyzer.detect()
        actions = analyzer.classify(include_context=True)
        profiles = analyzer.build_profiles()
    """

    def __init__(
        self,
        events: pl.DataFrame,
        config: GoalkeeperContextConfig | None = None,
    ):
        """
        Initialize the goalkeeper analyzer.

        Args:
            events: Full events DataFrame from SkillCorner data
            config: Optional configuration for context analysis
        """
        self.events = events
        self.config = config or GoalkeeperContextConfig()
        self._actions: pl.DataFrame | None = None

    def detect(self) -> pl.DataFrame:
        """Detect goalkeeper distribution events."""
        self._actions = detect_gk_actions(self.events)
        return self._actions

    def classify(self, include_context: bool = True) -> pl.DataFrame:
        """
        Classify distributions with optional context enrichment.

        Args:
            include_context: Whether to add transition speed and outcome context

        Returns:
            Classified actions DataFrame
        """
        if self._actions is None:
            self.detect()

        all_events = self.events if include_context else None
        self._actions = classify_gk_actions(self._actions, all_events=all_events)
        return self._actions

    def get_actions(self) -> pl.DataFrame:
        """Get the current actions DataFrame."""
        if self._actions is None:
            raise ValueError("No actions detected. Call detect() first.")
        return self._actions

    def build_profiles(self, min_distributions: int = 10) -> pl.DataFrame:
        """Build goalkeeper profiles from classified actions."""
        if self._actions is None:
            raise ValueError("No actions detected. Call detect() first.")
        profiles = build_goalkeeper_profiles(self._actions)
        return filter_goalkeeper_profiles(profiles, min_distributions)


def detect_gk_actions(events: pl.DataFrame) -> pl.DataFrame:
    """Detect goalkeeper distribution events."""
    return events.filter(
        (pl.col("player_position") == "GK") &
        (pl.col("event_type") == "player_possession")
    )


def add_distribution_context(
    actions: pl.DataFrame,
    all_events: pl.DataFrame,
    config: GoalkeeperContextConfig | None = None,
) -> pl.DataFrame:
    """
    Add transition speed context to goalkeeper distributions.

    For each distribution, calculates:
    - frames_from_phase_start: frames elapsed from possession phase start
    - distribution_speed_seconds: above converted to seconds (assuming 25fps)
    - is_quick_distribution: boolean if distribution within 3 seconds of phase start

    Args:
        actions: DataFrame of detected goalkeeper distributions
        all_events: Full events DataFrame with all possession events
        config: Optional configuration for thresholds

    Returns:
        actions DataFrame with added context columns
    """
    config = config or GoalkeeperContextConfig()

    if "phase_index" not in all_events.columns:
        return actions

    # Get phase start frames: minimum frame_start per phase per match
    phase_starts = all_events.group_by(
        ["match_id", "phase_index"]
    ).agg([
        pl.col("frame_start").min().alias("phase_frame_start")
    ])

    # Join phase start info to actions
    actions_with_context = actions.join(
        phase_starts,
        on=["match_id", "phase_index"],
        how="left"
    )

    # Calculate distribution timing metrics
    return actions_with_context.with_columns([
        # Frames from phase start to distribution
        (pl.col("frame_start") - pl.col("phase_frame_start")).alias("frames_from_phase_start"),
        # Convert to seconds (25 fps)
        ((pl.col("frame_start") - pl.col("phase_frame_start")) / config.fps).alias("distribution_speed_seconds"),
        # Quick distribution = within threshold
        ((pl.col("frame_start") - pl.col("phase_frame_start")) <= config.quick_distribution_threshold_frames).alias("is_quick_distribution"),
    ]).drop("phase_frame_start")


def add_outcome_labels(actions: pl.DataFrame) -> pl.DataFrame:
    """
    Add outcome labels for counter launches and attack outcomes.

    Args:
        actions: DataFrame of goalkeeper distributions

    Returns:
        actions DataFrame with is_counter_launch and led_to_attack columns
    """
    result = actions

    # Counter launch = distribution during transition or direct phase (fast attacks)
    if "team_in_possession_phase_type" in actions.columns:
        result = result.with_columns([
            pl.col("team_in_possession_phase_type").is_in(["transition", "direct"]).alias("is_counter_launch"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("is_counter_launch"),
        ])

    # Attack outcome = next phase is attacking (finish, create, direct)
    if "current_team_in_possession_next_phase_type" in actions.columns:
        result = result.with_columns([
            pl.col("current_team_in_possession_next_phase_type").is_in(
                ["finish", "create", "direct"]
            ).alias("led_to_attack"),
        ])
    elif "lead_to_shot" in actions.columns:
        # Fallback to lead_to_shot if next phase not available
        result = result.with_columns([
            pl.col("lead_to_shot").fill_null(False).alias("led_to_attack"),
        ])
    else:
        result = result.with_columns([
            pl.lit(False).alias("led_to_attack"),
        ])

    return result


def classify_gk_actions(
    actions: pl.DataFrame,
    all_events: pl.DataFrame | None = None,
    config: GoalkeeperContextConfig | None = None,
) -> pl.DataFrame:
    """
    Apply all goalkeeper action classifications.

    Args:
        actions: DataFrame of detected goalkeeper distributions
        all_events: Optional full events DataFrame for context (transition speed)
        config: Optional configuration for thresholds
    """
    result = add_outcome_labels(actions)

    # Add distribution context if full events provided
    if all_events is not None:
        result = add_distribution_context(result, all_events, config)

    return result


def build_goalkeeper_profiles(actions: pl.DataFrame) -> pl.DataFrame:
    """Build goalkeeper profiles from distribution events."""
    group_cols = ["player_id", "player_name", "team_id"]
    if "team_name" in actions.columns:
        group_cols.append("team_name")

    # Base aggregations
    agg_exprs = [
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

        # SkillCorner advanced metrics (only columns with data)
        pl.col("pass_ahead").mean().alias("pass_ahead_pct"),
        pl.col("player_targeted_xthreat").mean().alias("avg_targeted_xthreat"),
        pl.col("n_passing_options_dangerous_not_difficult").mean().alias("avg_safe_dangerous_options"),
        pl.col("forward_momentum").mean().alias("forward_momentum_pct"),
    ]

    # Add context metrics if available (from classify_gk_actions with all_events)
    if "distribution_speed_seconds" in actions.columns:
        agg_exprs.append(pl.col("distribution_speed_seconds").mean().alias("avg_distribution_speed"))

    if "is_counter_launch" in actions.columns:
        agg_exprs.append(pl.col("is_counter_launch").mean().alias("quick_counter_launch_pct"))

    if "led_to_attack" in actions.columns:
        agg_exprs.append(pl.col("led_to_attack").mean().alias("distribution_attack_rate"))

    profiles = actions.group_by(group_cols).agg(agg_exprs)

    # Convert to percentages
    pct_cols = [
        "pass_success_rate", "long_pass_pct", "short_pass_pct", "high_pass_pct",
        "quick_distribution_pct", "hand_pass_pct", "to_middle_third_pct",
        "to_attacking_third_pct", "pass_ahead_pct", "forward_momentum_pct",
        # New context metrics
        "quick_counter_launch_pct", "distribution_attack_rate",
    ]
    for col in pct_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns((pl.col(col) * 100).round(1).alias(col))

    # Round numeric columns
    numeric_cols = [
        "avg_pass_distance", "avg_passing_options", "avg_speed",
        "avg_targeted_xthreat", "avg_safe_dangerous_options",
        # New context metric
        "avg_distribution_speed",
    ]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles = profiles.with_columns(pl.col(col).round(2).alias(col))

    return profiles.sort("total_distributions", descending=True)


def filter_goalkeeper_profiles(profiles: pl.DataFrame, min_distributions: int = 10) -> pl.DataFrame:
    """Filter to goalkeepers with sufficient distributions."""
    return profiles.filter(pl.col("total_distributions") >= min_distributions)
