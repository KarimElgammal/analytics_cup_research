"""Defender mapper - StatsBomb to SkillCorner conversion."""

from __future__ import annotations

from src.archetypes.base import DEFENDER_FEATURE_NAMES
from src.archetypes.defenders import DEFENDER_COMPUTED, DEFENDER_WEIGHTS
from src.statsbomb.mappers.base import BaseMapper
from src.statsbomb.stats import DefenderStats


class DefenderMapper(BaseMapper):
    """Map defender stats to SkillCorner target profiles."""

    DISTRIBUTIONS = {
        "tackle_success_rate": {"p10": 30, "p25": 45, "p50": 55, "p75": 65, "p90": 80},
        "duel_success_rate": {"p10": 35, "p25": 45, "p50": 55, "p75": 65, "p90": 75},
        "aerial_success_rate": {"p10": 30, "p25": 45, "p50": 55, "p75": 70, "p90": 85},
        "pressure_success_rate": {"p10": 10, "p25": 20, "p50": 30, "p75": 40, "p90": 55},
        "pressures_per_90": {"p10": 5, "p25": 10, "p50": 15, "p75": 22, "p90": 30},
        "interceptions_per_90": {"p10": 0.5, "p25": 1.0, "p50": 1.5, "p75": 2.5, "p90": 4.0},
        "progressive_carry_pct": {"p10": 10, "p25": 20, "p50": 30, "p75": 45, "p90": 60},
    }

    COMPUTED_FEATURES = DEFENDER_COMPUTED
    FEATURE_NAMES = DEFENDER_FEATURE_NAMES

    def map_to_target(self, stats: DefenderStats) -> dict[str, float]:
        """Map defender stats to SkillCorner target profile.

        Mapping logic:
        - duel_success_rate → stop_danger_rate (winning duels stops attacks)
        - tackle_success_rate → reduce_danger_rate (tackles = danger reduction)
        - aerial_success_rate → goal_side_rate (aerial dominance = space control)
        - pressures_per_90 → pressing_rate (high activity = high pressing)
        """
        duel_pct = self.value_to_percentile(
            stats.duel_success_rate, self.DISTRIBUTIONS["duel_success_rate"]
        )
        aerial_pct = self.value_to_percentile(
            stats.aerial_success_rate, self.DISTRIBUTIONS["aerial_success_rate"]
        )

        return {
            # Mapped from StatsBomb data
            "stop_danger_rate": duel_pct,
            "reduce_danger_rate": self.value_to_percentile(
                stats.tackle_success_rate, self.DISTRIBUTIONS["tackle_success_rate"]
            ),
            "pressing_rate": self.value_to_percentile(
                stats.pressures_per_90, self.DISTRIBUTIONS["pressures_per_90"]
            ),
            "goal_side_rate": aerial_pct,
            # Beaten rates are inverse of success
            "beaten_by_possession_rate": max(15.0, 100 - duel_pct),
            "beaten_by_movement_rate": max(10.0, 100 - aerial_pct),
            # Engagement distance - estimate from pressing activity
            "avg_engagement_distance": max(
                5.0, min(95.0, 50 - (stats.pressures_per_90 / 30 * 20))
            ),
            # Force backward from tackle success
            "force_backward_rate": min(
                95.0,
                self.value_to_percentile(
                    stats.tackle_success_rate, self.DISTRIBUTIONS["tackle_success_rate"]
                )
                * 0.8,
            ),
        }

    def compute_weights(self, stats: DefenderStats) -> dict[str, float]:
        """Compute feature weights based on defender style."""
        weights = DEFENDER_WEIGHTS.copy()

        # High presser: boost pressing and engagement weights
        if stats.pressures_per_90 > 20:
            weights["pressing_rate"] += 0.10
            weights["avg_engagement_distance"] += 0.05

        # Aerial dominant: boost goal_side_rate
        if stats.aerial_success_rate > 65:
            weights["goal_side_rate"] += 0.08

        # Duel winner: boost stop_danger_rate
        if stats.duel_success_rate > 60:
            weights["stop_danger_rate"] += 0.08

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _compute_style_string(self, stats: DefenderStats) -> str:
        """Determine defensive style from stats."""
        notes = []

        if stats.pressures_per_90 > 20:
            notes.append("high pressing")
        elif stats.pressures_per_90 < 10:
            notes.append("positional")

        if stats.duel_success_rate > 60:
            notes.append("strong in duels")
        elif stats.duel_success_rate < 45:
            notes.append("sometimes beaten")

        if stats.aerial_success_rate > 65:
            notes.append("aerial presence")

        if stats.progressive_carry_pct > 35:
            notes.append("ball-playing")

        return ", ".join(notes) if notes else "balanced defender"

    def _get_summary_line(self, stats: DefenderStats) -> str:
        """Get summary line with match/event count."""
        event_count = stats.duels + stats.pressures
        return f"Computed from {stats.matches} matches ({event_count} events)."
