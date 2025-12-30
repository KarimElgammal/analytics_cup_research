"""Midfielder archetype definitions.

Midfielder archetypes are dynamically computed from StatsBomb API at runtime.
This module defines weights, directions, and the available archetype keys.

Source: StatsBomb open data via statsbombpy API
"""

from src.archetypes.base import Position

# Features computed from StatsBomb (vs estimated)
MIDFIELDER_COMPUTED = frozenset({
    "progressive_pass_pct",
    "progressive_carry_pct",
    "final_third_pass_pct",
    "pass_accuracy",
    "pressing_rate",
    "tackle_success_rate",
    "key_pass_rate",
    "through_ball_pct",
})

# Weights from GradientBoosting feature importance analysis
# Total = 1.0
MIDFIELDER_WEIGHTS: dict[str, float] = {
    # Ball Progression (0.30 total)
    "progressive_pass_pct": 0.12,
    "progressive_carry_pct": 0.08,
    "final_third_pass_pct": 0.06,
    "avg_pass_distance": 0.04,
    # Defensive Contribution (0.30 total)
    "pressing_rate": 0.10,
    "tackle_success_rate": 0.08,
    "interception_rate": 0.06,
    "ball_recovery_rate": 0.06,
    # Creativity & Chance Creation (0.25 total)
    "key_pass_rate": 0.10,
    "through_ball_pct": 0.08,
    "danger_creation_rate": 0.07,
    # Work Rate & Positioning (0.15 total)
    "central_presence_pct": 0.05,
    "attacking_third_pct": 0.05,
    "avg_speed": 0.05,
}

MIDFIELDER_DIRECTIONS: dict[str, int] = {
    # Ball Progression
    "progressive_pass_pct": 1,
    "progressive_carry_pct": 1,
    "final_third_pass_pct": 1,
    "avg_pass_distance": 0,  # Style indicator, not quality
    # Defensive
    "pressing_rate": 1,
    "tackle_success_rate": 1,
    "interception_rate": 1,
    "ball_recovery_rate": 1,
    # Creativity
    "key_pass_rate": 1,
    "through_ball_pct": 1,
    "danger_creation_rate": 1,
    # Work Rate
    "central_presence_pct": 0,  # Style indicator
    "attacking_third_pct": 0,  # Style indicator
    "avg_speed": 1,
    # Extra metrics that might be computed
    "pass_accuracy": 1,
}

# Available midfielder archetypes (dynamically loaded from StatsBomb)
MIDFIELDER_ARCHETYPE_OPTIONS: list[tuple[str, str]] = [
    ("Enzo Fernandez (ARG) - Box-to-box", "enzo"),
    ("Tchouaméni (FRA) - Defensive anchor", "tchouameni"),
    ("De Paul (ARG) - High work rate", "depaul"),
    ("Griezmann (FRA) - Creative CAM", "griezmann"),
    ("Pedri (ESP) - Technical control", "pedri"),
    ("Bellingham (ENG) - Complete midfielder", "bellingham"),
]
