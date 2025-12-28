"""Goalkeeper mapper - StatsBomb to SkillCorner conversion."""

from __future__ import annotations

from src.archetypes.base import GOALKEEPER_FEATURE_NAMES
from src.archetypes.goalkeepers import GOALKEEPER_COMPUTED, GOALKEEPER_WEIGHTS
from src.statsbomb.mappers.base import BaseMapper
from src.statsbomb.stats import GoalkeeperStats


class GoalkeeperMapper(BaseMapper):
    """Map goalkeeper stats to SkillCorner target profiles."""

    DISTRIBUTIONS = {
        "pass_success_rate": {"p10": 60, "p25": 68, "p50": 75, "p75": 82, "p90": 88},
        "avg_pass_distance": {"p10": 15, "p25": 22, "p50": 30, "p75": 38, "p90": 48},
        "long_pass_pct": {"p10": 15, "p25": 25, "p50": 40, "p75": 55, "p90": 70},
        "short_pass_pct": {"p10": 20, "p25": 35, "p50": 50, "p75": 65, "p90": 80},
        "high_pass_pct": {"p10": 10, "p25": 20, "p50": 35, "p75": 50, "p90": 65},
    }

    COMPUTED_FEATURES = GOALKEEPER_COMPUTED
    FEATURE_NAMES = GOALKEEPER_FEATURE_NAMES

    def map_to_target(self, stats: GoalkeeperStats) -> dict[str, float]:
        """Map goalkeeper stats to SkillCorner target profile.

        Mapping focuses on distribution style:
        - pass_success_rate → direct mapping
        - avg_pass_distance → direct mapping
        - long/short/high pass ratios → direct mapping
        """
        return {
            "pass_success_rate": self.value_to_percentile(
                stats.pass_success_rate, self.DISTRIBUTIONS["pass_success_rate"]
            ),
            "avg_pass_distance": self.value_to_percentile(
                stats.avg_pass_distance, self.DISTRIBUTIONS["avg_pass_distance"]
            ),
            "long_pass_pct": self.value_to_percentile(
                stats.long_pass_pct, self.DISTRIBUTIONS["long_pass_pct"]
            ),
            "short_pass_pct": self.value_to_percentile(
                stats.short_pass_pct, self.DISTRIBUTIONS["short_pass_pct"]
            ),
            "high_pass_pct": self.value_to_percentile(
                stats.high_pass_pct, self.DISTRIBUTIONS["high_pass_pct"]
            ),
            # Quick distribution - estimate from short pass tendency
            "quick_distribution_pct": max(
                30.0,
                self.value_to_percentile(
                    stats.short_pass_pct, self.DISTRIBUTIONS["short_pass_pct"]
                ),
            ),
            # To attacking third - estimate from long pass tendency
            "to_attacking_third_pct": self.value_to_percentile(
                stats.long_pass_pct, self.DISTRIBUTIONS["long_pass_pct"]
            )
            * 0.5,
        }

    def compute_weights(self, stats: GoalkeeperStats) -> dict[str, float]:
        """Compute feature weights based on distribution style."""
        weights = GOALKEEPER_WEIGHTS.copy()

        # Long distributor: boost long pass metrics
        if stats.long_pass_pct > 50:
            weights["long_pass_pct"] += 0.10
            weights["avg_pass_distance"] += 0.08
            weights["short_pass_pct"] -= 0.08

        # Sweeper keeper: boost short pass metrics
        if stats.short_pass_pct > 60:
            weights["short_pass_pct"] += 0.10
            weights["quick_distribution_pct"] += 0.05
            weights["long_pass_pct"] -= 0.08

        # Accurate distributor: boost pass success
        if stats.pass_success_rate > 80:
            weights["pass_success_rate"] += 0.08

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _compute_style_string(self, stats: GoalkeeperStats) -> str:
        """Determine distribution style from stats."""
        notes = []

        if stats.long_pass_pct > 50:
            notes.append("long distribution")
        elif stats.short_pass_pct > 60:
            notes.append("sweeper keeper")

        if stats.pass_success_rate > 80:
            notes.append("accurate")
        elif stats.pass_success_rate < 70:
            notes.append("direct")

        if stats.high_pass_pct > 45:
            notes.append("lofted passes")

        if stats.avg_pass_distance > 35:
            notes.append("deep launcher")

        return ", ".join(notes) if notes else "balanced goalkeeper"

    def _get_summary_line(self, stats: GoalkeeperStats) -> str:
        """Get summary line with match/distribution count."""
        return f"Computed from {stats.matches} matches ({stats.passes} distributions)."
