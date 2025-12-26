"""Map StatsBomb statistics to SkillCorner target profile values."""

from __future__ import annotations

from src.statsbomb.stats import PlayerStats


# Reference distributions for percentile mapping
# Based on typical forward statistics from football research
# Values represent approximate percentile thresholds
REFERENCE_DISTRIBUTIONS = {
    "conversion_rate": {
        "p10": 5.0,
        "p25": 8.0,
        "p50": 12.0,
        "p75": 18.0,
        "p90": 25.0,
    },
    "shot_accuracy": {
        "p10": 25.0,
        "p25": 35.0,
        "p50": 45.0,
        "p75": 55.0,
        "p90": 65.0,
    },
    "box_touches_per_90": {
        "p10": 2.0,
        "p25": 4.0,
        "p50": 7.0,
        "p75": 12.0,
        "p90": 18.0,
    },
    "pass_accuracy": {
        "p10": 65.0,
        "p25": 72.0,
        "p50": 80.0,
        "p75": 86.0,
        "p90": 92.0,
    },
    "dribble_success": {
        "p10": 35.0,
        "p25": 45.0,
        "p50": 55.0,
        "p75": 65.0,
        "p90": 75.0,
    },
    "key_passes_per_90": {
        "p10": 0.3,
        "p25": 0.6,
        "p50": 1.0,
        "p75": 1.8,
        "p90": 2.5,
    },
}


def _value_to_percentile(value: float, distribution: dict[str, float]) -> float:
    """Map a raw value to 0-100 percentile using reference distribution.

    Uses linear interpolation between percentile thresholds.

    Args:
        value: Raw statistic value
        distribution: Dict with p10, p25, p50, p75, p90 thresholds

    Returns:
        Percentile value between 0 and 100
    """
    if value <= 0:
        return 0.0

    p10 = distribution["p10"]
    p25 = distribution["p25"]
    p50 = distribution["p50"]
    p75 = distribution["p75"]
    p90 = distribution["p90"]

    # Interpolate between thresholds
    if value <= p10:
        return 10 * (value / p10) if p10 > 0 else 0
    elif value <= p25:
        return 10 + 15 * ((value - p10) / (p25 - p10)) if (p25 - p10) > 0 else 10
    elif value <= p50:
        return 25 + 25 * ((value - p25) / (p50 - p25)) if (p50 - p25) > 0 else 25
    elif value <= p75:
        return 50 + 25 * ((value - p50) / (p75 - p50)) if (p75 - p50) > 0 else 50
    elif value <= p90:
        return 75 + 15 * ((value - p75) / (p90 - p75)) if (p90 - p75) > 0 else 75
    else:
        # Above p90, extrapolate but cap at 100
        extra = 10 * ((value - p90) / p90) if p90 > 0 else 0
        return min(100.0, 90 + extra)


def map_to_skillcorner_target(stats: PlayerStats) -> dict[str, float]:
    """Map StatsBomb player stats to SkillCorner target profile.

    The mapping translates event-based metrics to tracking-equivalent targets:
    - conversion_rate → danger_rate (clinical finishing = dangerous entries)
    - box_touches_per_90 → avg_separation (finds space in box = good separation)
    - pass_accuracy → avg_passing_options (link-up play)
    - dribble_success → carry_pct (dribbling ability)

    For tracking-specific metrics without StatsBomb equivalents, we use
    reasonable default values based on position (forward).

    Args:
        stats: PlayerStats computed from StatsBomb data

    Returns:
        Dictionary with SkillCorner target profile values (0-100 scale)
    """
    return {
        # Mapped from StatsBomb data
        "danger_rate": _value_to_percentile(
            stats.conversion_rate,
            REFERENCE_DISTRIBUTIONS["conversion_rate"]
        ),
        "avg_separation": _value_to_percentile(
            stats.box_touches_per_90,
            REFERENCE_DISTRIBUTIONS["box_touches_per_90"]
        ),
        "avg_passing_options": _value_to_percentile(
            stats.pass_accuracy,
            REFERENCE_DISTRIBUTIONS["pass_accuracy"]
        ),
        "carry_pct": _value_to_percentile(
            stats.dribble_success,
            REFERENCE_DISTRIBUTIONS["dribble_success"]
        ),

        # Fixed values for tracking-specific metrics (no StatsBomb equivalent)
        # These represent reasonable forward archetypes
        "central_pct": 70.0,           # Forwards typically central
        "avg_entry_speed": 65.0,       # Moderate speed (not pace-reliant)
        "half_space_pct": 55.0,        # Some half-space movement
        "quick_break_pct": 50.0,       # Moderate counter-attack involvement
        "avg_defensive_line_dist": 50.0,  # Average positioning depth
        "avg_teammates_ahead": 40.0,   # Forward = fewer ahead
        "avg_distance": 50.0,          # Average work rate
        "goal_rate": 50.0,             # Average goal conversion on entries
    }


def get_archetype_description(stats: PlayerStats, target: dict[str, float]) -> str:
    """Generate human-readable description of the archetype.

    Args:
        stats: Original PlayerStats from StatsBomb
        target: Mapped target profile

    Returns:
        Multi-line description string
    """
    # Determine style based on key metrics
    style_notes = []

    if stats.conversion_rate > 18:
        style_notes.append("clinical finisher")
    elif stats.conversion_rate < 10:
        style_notes.append("volume shooter")

    if stats.dribble_success > 60:
        style_notes.append("skilled dribbler")
    elif stats.dribble_success < 45:
        style_notes.append("movement-focused (not a dribbler)")

    if stats.box_touches_per_90 > 10:
        style_notes.append("frequent box presence")

    if stats.key_passes_per_90 > 1.5:
        style_notes.append("creative passer")

    style_str = ", ".join(style_notes) if style_notes else "balanced forward"

    return f"""Archetype computed from StatsBomb event data:
- {stats.matches} matches, ~{stats.minutes:.0f} minutes analyzed
- {stats.shot_accuracy:.1f}% shot accuracy, {stats.conversion_rate:.1f}% conversion rate
- {stats.goals} goals from {stats.shots} shots
- {stats.box_touches} box touches ({stats.box_touches_per_90:.1f} per 90)
- {stats.pass_accuracy:.1f}% pass accuracy, {stats.key_passes} key passes
- {stats.dribble_success:.1f}% dribble success rate

Style: {style_str}

Target profile (0-100 scale):
- danger_rate: {target['danger_rate']:.0f}
- avg_separation: {target['avg_separation']:.0f}
- carry_pct: {target['carry_pct']:.0f}
"""
