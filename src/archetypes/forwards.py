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
    "carry_pct",
    "avg_passing_options",
})

# Weights from one-time GradientBoosting feature importance analysis
FORWARD_WEIGHTS: dict[str, float] = {
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

FORWARD_DIRECTIONS: dict[str, int] = {
    "danger_rate": 1,
    "avg_separation": 1,
    "carry_pct": 1,
    "central_pct": 1,
    "avg_entry_speed": 1,
    "avg_passing_options": 1,
    "half_space_pct": 1,
    "quick_break_pct": 1,
    "avg_defensive_line_dist": 1,
    "avg_teammates_ahead": -1,
    "goal_rate": 1,
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
