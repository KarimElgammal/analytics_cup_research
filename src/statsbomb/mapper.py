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
        # Floor at 30 - limited StatsBomb data shouldn't produce 0 targets
        "carry_pct": max(30.0, _value_to_percentile(
            stats.dribble_success,
            REFERENCE_DISTRIBUTIONS["dribble_success"]
        )),

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


def compute_archetype_weights(stats: PlayerStats) -> dict[str, float]:
    """Compute feature weights from StatsBomb stats.

    Weights are derived directly from the player's actual statistics,
    making each archetype data-driven rather than manually defined.

    Mapping logic:
    - dribble_success → carry_pct weight (dribblers weight this higher)
    - conversion_rate → danger_rate weight (clinical finishers)
    - box_touches_per_90 → avg_separation weight (movement players)
    - key_passes_per_90 → avg_passing_options weight (creative players)

    Args:
        stats: PlayerStats computed from StatsBomb data

    Returns:
        Dictionary with feature weights summing to 1.0
    """
    # Convert stats to 0-1 scale using reference distributions
    dribble_score = _value_to_percentile(
        stats.dribble_success, REFERENCE_DISTRIBUTIONS["dribble_success"]
    ) / 100.0
    conversion_score = _value_to_percentile(
        stats.conversion_rate, REFERENCE_DISTRIBUTIONS["conversion_rate"]
    ) / 100.0
    box_touch_score = _value_to_percentile(
        stats.box_touches_per_90, REFERENCE_DISTRIBUTIONS["box_touches_per_90"]
    ) / 100.0
    key_pass_score = _value_to_percentile(
        stats.key_passes_per_90, REFERENCE_DISTRIBUTIONS["key_passes_per_90"]
    ) / 100.0

    # Base weights (static, from one-time GradientBoosting analysis)
    # These get adjusted based on player stats
    base_weights = {
        "avg_separation": 0.20,
        "danger_rate": 0.18,
        "avg_entry_speed": 0.15,
        "avg_defensive_line_dist": 0.12,
        "central_pct": 0.10,
        "carry_pct": 0.08,
        "avg_passing_options": 0.08,
        "quick_break_pct": 0.04,
        "avg_teammates_ahead": 0.03,
        "half_space_pct": 0.02,
    }

    # Adjust weights based on player stats
    weights = base_weights.copy()

    # Dribblers: increase carry_pct and speed, decrease separation
    if dribble_score > 0.5:  # Above average dribbler
        dribble_boost = (dribble_score - 0.5) * 0.4  # Up to +0.20
        weights["carry_pct"] += dribble_boost
        weights["avg_entry_speed"] += dribble_boost * 0.5
        weights["avg_separation"] -= dribble_boost * 0.5  # Dribblers run WITH ball

    # Clinical finishers: increase danger_rate
    if conversion_score > 0.6:  # Above average finisher
        conversion_boost = (conversion_score - 0.6) * 0.3  # Up to +0.12
        weights["danger_rate"] += conversion_boost

    # Movement players: increase separation (high box touches = finds space)
    if box_touch_score > 0.7:  # High box presence
        movement_boost = (box_touch_score - 0.7) * 0.3  # Up to +0.09
        weights["avg_separation"] += movement_boost

    # Creative players: increase passing options
    if key_pass_score > 0.5:  # Creative
        creative_boost = (key_pass_score - 0.5) * 0.3  # Up to +0.15
        weights["avg_passing_options"] += creative_boost

    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


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

Target profile (percentile targets 0-100):
| Feature | Target | Source |
|---------|--------|--------|
| Danger Rate | {target['danger_rate']:.0f} | Computed from conversion rate |
| Separation | {target['avg_separation']:.0f} | Computed from box touches |
| Carry % | {target['carry_pct']:.0f} | Computed from dribble success |
| Central % | {target['central_pct']:.0f} | Guessed (no tracking data) |
| Speed | {target['avg_entry_speed']:.0f} | Guessed (no tracking data) |

How radar values are calculated:
Player values show their percentile rank across all A-League players analyzed.
Example: A player at 60 on "Central %" is at the 60th percentile (better than 60% of players).
"""


# === DEFENDER MAPPING ===

DEFENDER_DISTRIBUTIONS = {
    "tackle_success_rate": {"p10": 30, "p25": 45, "p50": 55, "p75": 65, "p90": 80},
    "duel_success_rate": {"p10": 35, "p25": 45, "p50": 55, "p75": 65, "p90": 75},
    "aerial_success_rate": {"p10": 30, "p25": 45, "p50": 55, "p75": 70, "p90": 85},
    "pressure_success_rate": {"p10": 10, "p25": 20, "p50": 30, "p75": 40, "p90": 55},
    "pressures_per_90": {"p10": 5, "p25": 10, "p50": 15, "p75": 22, "p90": 30},
    "interceptions_per_90": {"p10": 0.5, "p25": 1.0, "p50": 1.5, "p75": 2.5, "p90": 4.0},
    "progressive_carry_pct": {"p10": 10, "p25": 20, "p50": 30, "p75": 45, "p90": 60},
}


def map_defender_to_skillcorner(stats) -> dict[str, float]:
    """Map defender stats to SkillCorner target profile.

    Mapping logic:
    - duel_success_rate → stop_danger_rate (winning duels stops attacks)
    - pressure_success_rate → reduce_danger_rate (successful pressure = danger reduction)
    - aerial_success_rate → goal_side_rate (aerial dominance = controlling space)
    - pressures_per_90 → pressing_rate (high activity = high pressing)
    - duel losses → beaten rates (inverse of success)

    Args:
        stats: DefenderStats computed from StatsBomb data

    Returns:
        Dictionary with SkillCorner target profile values (0-100 scale)
    """
    from src.statsbomb.stats import DefenderStats

    return {
        # Mapped from StatsBomb data
        "stop_danger_rate": _value_to_percentile(
            stats.duel_success_rate, DEFENDER_DISTRIBUTIONS["duel_success_rate"]
        ),
        "reduce_danger_rate": _value_to_percentile(
            stats.tackle_success_rate, DEFENDER_DISTRIBUTIONS["tackle_success_rate"]
        ),
        "pressing_rate": _value_to_percentile(
            stats.pressures_per_90, DEFENDER_DISTRIBUTIONS["pressures_per_90"]
        ),
        "goal_side_rate": _value_to_percentile(
            stats.aerial_success_rate, DEFENDER_DISTRIBUTIONS["aerial_success_rate"]
        ),
        # Beaten rates are inverse (lower is better in duels = higher beaten rate)
        "beaten_by_possession_rate": max(15.0, 100 - _value_to_percentile(
            stats.duel_success_rate, DEFENDER_DISTRIBUTIONS["duel_success_rate"]
        )),
        "beaten_by_movement_rate": max(10.0, 100 - _value_to_percentile(
            stats.aerial_success_rate, DEFENDER_DISTRIBUTIONS["aerial_success_rate"]
        )),
        # Engagement distance - guess based on pressing activity (clamped to 5-95)
        "avg_engagement_distance": max(5.0, min(95.0, 50 - (stats.pressures_per_90 / 30 * 20))),
        # Force backward from tackle success
        "force_backward_rate": min(95.0, _value_to_percentile(
            stats.tackle_success_rate, DEFENDER_DISTRIBUTIONS["tackle_success_rate"]
        ) * 0.8),
    }


def get_defender_description(stats, target: dict[str, float]) -> str:
    """Generate description for defender archetype.

    Args:
        stats: DefenderStats from StatsBomb
        target: Mapped target profile

    Returns:
        Multi-line description string
    """
    style_notes = []

    if stats.pressures_per_90 > 20:
        style_notes.append("high pressing intensity")
    elif stats.pressures_per_90 < 10:
        style_notes.append("positional defender")

    if stats.duel_success_rate > 60:
        style_notes.append("dominant in duels")
    elif stats.duel_success_rate < 45:
        style_notes.append("sometimes beaten in 1v1")

    if stats.aerial_success_rate > 65:
        style_notes.append("aerial presence")

    if stats.progressive_carry_pct > 35:
        style_notes.append("ball-playing defender")

    style_str = ", ".join(style_notes) if style_notes else "balanced defender"

    return f"""Archetype computed from StatsBomb event data:
- {stats.matches} matches analyzed
- {stats.tackles} tackles ({stats.tackle_success_rate:.1f}% success)
- {stats.duels} duels ({stats.duel_success_rate:.1f}% won)
- {stats.aerial_duels} aerial duels ({stats.aerial_success_rate:.1f}% won)
- {stats.pressures} pressures ({stats.pressures_per_90:.1f} per 90)
- {stats.interceptions} interceptions, {stats.clearances} clearances

Style: {style_str}

Target profile (percentile targets 0-100):
| Feature | Target | Source |
|---------|--------|--------|
| Stop Danger % | {target['stop_danger_rate']:.0f} | Computed from duel success |
| Reduce Danger % | {target['reduce_danger_rate']:.0f} | Computed from tackle success |
| Pressing % | {target['pressing_rate']:.0f} | Computed from pressures/90 |
| Goal Side % | {target['goal_side_rate']:.0f} | Computed from aerial success |
| Beaten (Ball) % | {target['beaten_by_possession_rate']:.0f} | Inverse of duel success |
"""


def compute_defender_weights(stats) -> dict[str, float]:
    """Compute feature weights for defender archetype.

    Args:
        stats: DefenderStats from StatsBomb

    Returns:
        Dictionary with feature weights summing to 1.0
    """
    # Base weights
    base_weights = {
        "stop_danger_rate": 0.20,
        "reduce_danger_rate": 0.15,
        "pressing_rate": 0.15,
        "goal_side_rate": 0.15,
        "avg_engagement_distance": 0.10,
        "beaten_by_possession_rate": 0.10,
        "beaten_by_movement_rate": 0.08,
        "force_backward_rate": 0.07,
    }

    weights = base_weights.copy()

    # Adjust based on player style
    if stats.pressures_per_90 > 20:  # High presser
        weights["pressing_rate"] += 0.10
        weights["avg_engagement_distance"] += 0.05

    if stats.aerial_success_rate > 65:  # Aerial dominant
        weights["goal_side_rate"] += 0.08

    if stats.duel_success_rate > 60:  # Duel winner
        weights["stop_danger_rate"] += 0.08

    # Normalize
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# === GOALKEEPER MAPPING ===

GK_DISTRIBUTIONS = {
    "pass_success_rate": {"p10": 60, "p25": 68, "p50": 75, "p75": 82, "p90": 88},
    "avg_pass_distance": {"p10": 15, "p25": 22, "p50": 30, "p75": 38, "p90": 48},
    "long_pass_pct": {"p10": 15, "p25": 25, "p50": 40, "p75": 55, "p90": 70},
    "short_pass_pct": {"p10": 20, "p25": 35, "p50": 50, "p75": 65, "p90": 80},
    "high_pass_pct": {"p10": 10, "p25": 20, "p50": 35, "p75": 50, "p90": 65},
}


def map_goalkeeper_to_skillcorner(stats) -> dict[str, float]:
    """Map goalkeeper stats to SkillCorner target profile.

    Args:
        stats: GoalkeeperStats computed from StatsBomb data

    Returns:
        Dictionary with SkillCorner target profile values (0-100 scale)
    """
    return {
        "pass_success_rate": _value_to_percentile(
            stats.pass_success_rate, GK_DISTRIBUTIONS["pass_success_rate"]
        ),
        "avg_pass_distance": _value_to_percentile(
            stats.avg_pass_distance, GK_DISTRIBUTIONS["avg_pass_distance"]
        ),
        "long_pass_pct": _value_to_percentile(
            stats.long_pass_pct, GK_DISTRIBUTIONS["long_pass_pct"]
        ),
        "short_pass_pct": _value_to_percentile(
            stats.short_pass_pct, GK_DISTRIBUTIONS["short_pass_pct"]
        ),
        "high_pass_pct": _value_to_percentile(
            stats.high_pass_pct, GK_DISTRIBUTIONS["high_pass_pct"]
        ),
        # Quick distribution - estimate from short pass tendency
        "quick_distribution_pct": max(30.0, _value_to_percentile(
            stats.short_pass_pct, GK_DISTRIBUTIONS["short_pass_pct"]
        )),
        # To attacking third - estimate from long pass tendency
        "to_attacking_third_pct": _value_to_percentile(
            stats.long_pass_pct, GK_DISTRIBUTIONS["long_pass_pct"]
        ) * 0.5,
    }


def get_goalkeeper_description(stats, target: dict[str, float]) -> str:
    """Generate description for goalkeeper archetype.

    Args:
        stats: GoalkeeperStats from StatsBomb
        target: Mapped target profile

    Returns:
        Multi-line description string
    """
    style_notes = []

    if stats.long_pass_pct > 50:
        style_notes.append("long distribution specialist")
    elif stats.short_pass_pct > 60:
        style_notes.append("sweeper keeper (short passing)")

    if stats.pass_success_rate > 80:
        style_notes.append("accurate distributor")
    elif stats.pass_success_rate < 70:
        style_notes.append("direct but risky")

    if stats.high_pass_pct > 45:
        style_notes.append("frequent lofted passes")

    if stats.avg_pass_distance > 35:
        style_notes.append("launches attacks from deep")

    style_str = ", ".join(style_notes) if style_notes else "balanced goalkeeper"

    return f"""Archetype computed from StatsBomb event data:
- {stats.matches} matches analyzed
- {stats.saves} saves, {stats.goals_conceded} goals conceded
- {stats.passes} distributions ({stats.pass_success_rate:.1f}% success)
- {stats.avg_pass_distance:.1f}m average pass distance
- {stats.long_pass_pct:.1f}% long passes, {stats.short_pass_pct:.1f}% short passes
- {stats.high_pass_pct:.1f}% high/lofted passes

Style: {style_str}

Target profile (percentile targets 0-100):
| Feature | Target | Source |
|---------|--------|--------|
| Pass Success % | {target['pass_success_rate']:.0f} | Computed from pass accuracy |
| Pass Distance | {target['avg_pass_distance']:.0f} | Computed from avg distance |
| Long Pass % | {target['long_pass_pct']:.0f} | Computed from long pass ratio |
| Short Pass % | {target['short_pass_pct']:.0f} | Computed from short pass ratio |
| High Pass % | {target['high_pass_pct']:.0f} | Computed from lofted passes |
"""


def compute_goalkeeper_weights(stats) -> dict[str, float]:
    """Compute feature weights for goalkeeper archetype.

    Args:
        stats: GoalkeeperStats from StatsBomb

    Returns:
        Dictionary with feature weights summing to 1.0
    """
    base_weights = {
        "pass_success_rate": 0.25,
        "avg_pass_distance": 0.15,
        "long_pass_pct": 0.15,
        "short_pass_pct": 0.15,
        "high_pass_pct": 0.10,
        "quick_distribution_pct": 0.10,
        "to_attacking_third_pct": 0.10,
    }

    weights = base_weights.copy()

    # Adjust based on style
    if stats.long_pass_pct > 50:  # Long distributor
        weights["long_pass_pct"] += 0.10
        weights["avg_pass_distance"] += 0.08
        weights["short_pass_pct"] -= 0.08

    if stats.short_pass_pct > 60:  # Sweeper keeper
        weights["short_pass_pct"] += 0.10
        weights["quick_distribution_pct"] += 0.05
        weights["long_pass_pct"] -= 0.08

    if stats.pass_success_rate > 80:  # Accurate
        weights["pass_success_rate"] += 0.08

    # Normalize
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
