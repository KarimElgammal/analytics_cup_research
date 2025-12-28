"""Defender archetype definitions.

Archetypes computed from StatsBomb open data (World Cup 2022, Euro 2024).
Source: https://github.com/statsbomb/open-data
Script: scripts/compute_archetype_profiles.py
"""

from src.archetypes.base import ArchetypeConfig, Position

# Features computed from StatsBomb (vs estimated)
DEFENDER_COMPUTED = frozenset({
    "stop_danger_rate",
    "reduce_danger_rate",
    "pressing_rate",
    "goal_side_rate",
    "beaten_by_movement_rate",
    "avg_engagement_distance",
})

# Weights derived from one-time ML analysis (GradientBoosting, AUC 0.845)
DEFENDER_WEIGHTS: dict[str, float] = {
    "stop_danger_rate": 0.30,
    "avg_engagement_distance": 0.17,
    "reduce_danger_rate": 0.15,
    "beaten_by_possession_rate": 0.15,
    "beaten_by_movement_rate": 0.08,
    "force_backward_rate": 0.08,
    "pressing_rate": 0.04,
    "goal_side_rate": 0.03,
}

DEFENDER_DIRECTIONS: dict[str, int] = {
    "stop_danger_rate": 1,
    "reduce_danger_rate": 1,
    "force_backward_rate": 1,
    "beaten_by_possession_rate": -1,
    "beaten_by_movement_rate": -1,
    "pressing_rate": 1,
    "goal_side_rate": 1,
    "avg_engagement_distance": -1,
    "defensive_third_pct": 1,
    "middle_third_pct": 0,
    "attacking_third_pct": -1,
}

# Archetype definitions
DEFENDER_ARCHETYPES: dict[str, ArchetypeConfig] = {
    "gvardiol": ArchetypeConfig(
        key="gvardiol",
        name="Josko Gvardiol (CRO) - Ball-playing CB",
        description="Balanced pressing, strong in duels, deeper engagement position.",
        position=Position.DEFENDER,
        computed_features=DEFENDER_COMPUTED,
        source_matches=3,
        source_events=48,
        target_profile={
            "stop_danger_rate": 67,
            "reduce_danger_rate": 32,
            "force_backward_rate": 50,
            "beaten_by_possession_rate": 20,
            "beaten_by_movement_rate": 33,
            "pressing_rate": 50,
            "goal_side_rate": 62,
            "avg_engagement_distance": 32,
            "middle_third_pct": 40,
        },
    ),
    "vandijk": ArchetypeConfig(
        key="vandijk",
        name="Virgil van Dijk (NED) - Commanding CB",
        description="Excellent positional defender who reads the game. Minimal duels - wins through positioning.",
        position=Position.DEFENDER,
        computed_features=DEFENDER_COMPUTED,
        source_matches=3,
        source_events=42,
        target_profile={
            "stop_danger_rate": 50,
            "reduce_danger_rate": 17,
            "force_backward_rate": 60,
            "beaten_by_possession_rate": 15,
            "beaten_by_movement_rate": 20,
            "pressing_rate": 36,
            "goal_side_rate": 90,
            "avg_engagement_distance": 32,
            "defensive_third_pct": 60,
        },
    ),
    "hakimi": ArchetypeConfig(
        key="hakimi",
        name="Achraf Hakimi (MAR) - Attacking Wing-back",
        description="High pressing, aggressive engagement, plays higher up the pitch.",
        position=Position.DEFENDER,
        computed_features=DEFENDER_COMPUTED,
        source_matches=6,
        source_events=148,
        target_profile={
            "stop_danger_rate": 41,
            "reduce_danger_rate": 25,
            "force_backward_rate": 45,
            "beaten_by_possession_rate": 30,
            "beaten_by_movement_rate": 59,
            "pressing_rate": 59,
            "goal_side_rate": 42,
            "avg_engagement_distance": 35,
            "middle_third_pct": 45,
            "attacking_third_pct": 25,
        },
    ),
}
