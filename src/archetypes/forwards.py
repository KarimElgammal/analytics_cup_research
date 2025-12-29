"""Forward archetype definitions.

Forward archetypes are dynamically computed from StatsBomb API at runtime.
This module defines weights, directions, and the available archetype keys.

Source: StatsBomb open data via statsbombpy API
"""

from src.archetypes.base import Position

# Features computed from StatsBomb (vs estimated)
FORWARD_COMPUTED = frozenset({
    "danger_rate",
    "avg_separation",
    "avg_passing_options",
})

# Weights from one-time GradientBoosting feature importance analysis
# Rebalanced to include transition speed and passer/receiver metrics
FORWARD_WEIGHTS: dict[str, float] = {
    # Core metrics (reduced to make room for new metrics)
    "avg_separation": 0.18,
    "danger_rate": 0.16,
    "avg_entry_speed": 0.12,
    "avg_defensive_line_dist": 0.10,
    "central_pct": 0.09,
    "avg_passing_options": 0.07,
    "quick_break_pct": 0.03,
    "avg_teammates_ahead": 0.02,
    "half_space_pct": 0.02,
    # Transition speed metrics (new)
    "avg_transition_speed": 0.04,
    "fast_transition_pct": 0.04,
    # Passer/Receiver credit metrics (new)
    "assisted_danger_rate": 0.05,
    "solo_danger_rate": 0.04,
    "assist_danger_rate": 0.04,
}

FORWARD_DIRECTIONS: dict[str, int] = {
    "danger_rate": 1,
    "avg_separation": 1,
    "central_pct": 1,
    "avg_entry_speed": 1,
    "avg_passing_options": 1,
    "half_space_pct": 1,
    "quick_break_pct": 1,
    "avg_defensive_line_dist": 1,
    "avg_teammates_ahead": -1,
    "goal_rate": 1,
    # Transition speed (lower = faster = better)
    "avg_transition_speed": -1,
    "fast_transition_pct": 1,
    # Passer/Receiver metrics
    "assisted_pct": 0,  # Neutral - style indicator, not quality
    "assisted_danger_rate": 1,
    "solo_danger_rate": 1,
    "total_entry_assists": 1,
    "assist_danger_rate": 1,
}

# Available forward archetypes (dynamically loaded from StatsBomb)
FORWARD_ARCHETYPE_OPTIONS: list[tuple[str, str]] = [
    ("Alvarez (ARG) - Movement-focused", "alvarez"),
    ("Giroud (FRA) - Target man", "giroud"),
    ("Kane (ENG) - Complete forward", "kane"),
    ("Lewandowski (POL) - Clinical poacher", "lewandowski"),
    ("Rashford (ENG) - Pace/dribbling", "rashford"),
    ("En-Nesyri (MAR) - Physical forward", "en_nesyri"),
]
