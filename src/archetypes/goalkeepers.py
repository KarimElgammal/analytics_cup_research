"""Goalkeeper archetype definitions.

Archetypes computed from StatsBomb open data (World Cup 2022, Euro 2024).
Source: https://github.com/statsbomb/open-data
Script: scripts/compute_archetype_profiles.py
"""

from src.archetypes.base import ArchetypeConfig, Position

# Features computed from StatsBomb (vs estimated)
GOALKEEPER_COMPUTED = frozenset({
    "pass_success_rate",
    "long_pass_pct",
    "avg_pass_distance",
})

# Base weights derived from one-time ML analysis (GradientBoosting, AUC 0.993)
# Balanced for style comparison (ML found pass_distance dominates at 98.6%)
# Rebalanced Dec 2025 to include distribution context metrics
GOALKEEPER_WEIGHTS: dict[str, float] = {
    # Core metrics (reduced to make room for new metrics)
    "pass_success_rate": 0.18,
    "avg_pass_distance": 0.18,
    "long_pass_pct": 0.13,
    "short_pass_pct": 0.13,
    "quick_distribution_pct": 0.08,
    "to_middle_third_pct": 0.08,
    "avg_passing_options": 0.08,
    # Distribution context metrics (new Dec 2025)
    "avg_distribution_speed": 0.05,
    "quick_counter_launch_pct": 0.05,
    "distribution_attack_rate": 0.04,
}

GOALKEEPER_DIRECTIONS: dict[str, int] = {
    "pass_success_rate": 1,
    "long_pass_pct": 1,
    "short_pass_pct": 1,
    "avg_pass_distance": 1,
    "quick_distribution_pct": 1,
    "high_pass_pct": 1,
    "to_middle_third_pct": 1,
    "to_attacking_third_pct": 1,
    "avg_passing_options": 1,
    "hand_pass_pct": -1,
    # Distribution context metrics (new Dec 2025)
    "avg_distribution_speed": -1,  # Lower = faster = better
    "quick_counter_launch_pct": 1,
    "distribution_attack_rate": 1,
}

# Style-specific weight/direction adjustments
# These are merged with base weights when creating archetypes

NEUER_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
NEUER_WEIGHTS["pass_success_rate"] = 0.25
NEUER_WEIGHTS["long_pass_pct"] = 0.15
NEUER_WEIGHTS["avg_pass_distance"] = 0.15

NEUER_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()

LLORIS_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
LLORIS_WEIGHTS["short_pass_pct"] = 0.20
LLORIS_WEIGHTS["long_pass_pct"] = 0.05

LLORIS_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()
LLORIS_DIRECTIONS["short_pass_pct"] = 1
LLORIS_DIRECTIONS["long_pass_pct"] = -1

BOUNOU_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
BOUNOU_WEIGHTS["long_pass_pct"] = 0.20
BOUNOU_WEIGHTS["avg_pass_distance"] = 0.20
BOUNOU_WEIGHTS["high_pass_pct"] = 0.15
BOUNOU_WEIGHTS["short_pass_pct"] = 0.00

BOUNOU_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()
BOUNOU_DIRECTIONS["long_pass_pct"] = 1
BOUNOU_DIRECTIONS["avg_pass_distance"] = 1
BOUNOU_DIRECTIONS["high_pass_pct"] = 1
BOUNOU_DIRECTIONS["short_pass_pct"] = -1


# Archetype definitions with their custom weights/directions
GOALKEEPER_ARCHETYPES: dict[str, tuple[ArchetypeConfig, dict[str, float], dict[str, int]]] = {
    "neuer": (
        ArchetypeConfig(
            key="neuer",
            name="Manuel Neuer (GER) - Modern Sweeper",
            description="90% pass accuracy, 46m average, 25% long balls. Excellent accuracy with balanced distribution.",
            position=Position.GOALKEEPER,
            computed_features=GOALKEEPER_COMPUTED,
            source_matches=2,
            source_events=0,
            target_profile={
                "pass_success_rate": 90,
                "long_pass_pct": 25,
                "short_pass_pct": 45,
                "avg_pass_distance": 46,
                "quick_distribution_pct": 50,
                "high_pass_pct": 30,
            },
        ),
        NEUER_WEIGHTS,
        NEUER_DIRECTIONS,
    ),
    "lloris": (
        ArchetypeConfig(
            key="lloris",
            name="Hugo Lloris (FRA) - Direct Distributor",
            description="55% pass accuracy, 83m average, 69% long balls. France's direct style bypasses midfield.",
            position=Position.GOALKEEPER,
            computed_features=GOALKEEPER_COMPUTED,
            source_matches=2,
            source_events=0,
            target_profile={
                "pass_success_rate": 55,
                "long_pass_pct": 69,
                "short_pass_pct": 20,
                "avg_pass_distance": 83,
                "quick_distribution_pct": 40,
                "high_pass_pct": 60,
            },
        ),
        LLORIS_WEIGHTS,
        LLORIS_DIRECTIONS,
    ),
    "bounou": (
        ArchetypeConfig(
            key="bounou",
            name="Yassine Bounou (MAR) - Balanced Long Distributor",
            description="72% pass accuracy, 70m average, 50% long balls. Morocco's reliable distribution.",
            position=Position.GOALKEEPER,
            computed_features=GOALKEEPER_COMPUTED,
            source_matches=5,
            source_events=0,
            target_profile={
                "pass_success_rate": 72,
                "long_pass_pct": 50,
                "short_pass_pct": 30,
                "avg_pass_distance": 70,
                "quick_distribution_pct": 45,
                "high_pass_pct": 45,
            },
        ),
        BOUNOU_WEIGHTS,
        BOUNOU_DIRECTIONS,
    ),
}
