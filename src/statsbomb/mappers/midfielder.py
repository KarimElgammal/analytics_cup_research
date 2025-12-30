"""Midfielder mapper - StatsBomb to SkillCorner conversion."""

from __future__ import annotations

from src.archetypes.base import MIDFIELDER_FEATURE_NAMES
from src.archetypes.midfielders import MIDFIELDER_COMPUTED, MIDFIELDER_WEIGHTS
from src.statsbomb.mappers.base import BaseMapper
from src.statsbomb.stats import MidfielderStats


class MidfielderMapper(BaseMapper):
    """Map midfielder stats to SkillCorner target profiles."""

    DISTRIBUTIONS = {
        "pass_accuracy": {"p10": 70, "p25": 78, "p50": 84, "p75": 89, "p90": 93},
        "progressive_pass_pct": {"p10": 5, "p25": 10, "p50": 15, "p75": 22, "p90": 30},
        "progressive_carry_pct": {"p10": 8, "p25": 15, "p50": 22, "p75": 32, "p90": 45},
        "final_third_pass_pct": {"p10": 10, "p25": 18, "p50": 25, "p75": 35, "p90": 45},
        "pressures_per_90": {"p10": 8, "p25": 14, "p50": 20, "p75": 28, "p90": 38},
        "tackle_success_rate": {"p10": 35, "p25": 48, "p50": 58, "p75": 68, "p90": 80},
        "interceptions_per_90": {"p10": 0.5, "p25": 1.0, "p50": 1.8, "p75": 2.8, "p90": 4.0},
        "ball_recoveries_per_90": {"p10": 2, "p25": 4, "p50": 6, "p75": 9, "p90": 13},
        "key_pass_rate": {"p10": 0.5, "p25": 1.2, "p50": 2.0, "p75": 3.2, "p90": 5.0},
        "through_ball_pct": {"p10": 0.2, "p25": 0.8, "p50": 1.5, "p75": 2.8, "p90": 4.5},
    }

    COMPUTED_FEATURES = MIDFIELDER_COMPUTED
    FEATURE_NAMES = MIDFIELDER_FEATURE_NAMES

    def map_to_target(self, stats: MidfielderStats) -> dict[str, float]:
        """Map midfielder stats to SkillCorner target profile.

        Mapping logic:
        - progressive_pass_pct → ball progression indicator
        - progressive_carry_pct → ball carrying threat
        - pressures_per_90 → pressing intensity
        - tackle_success_rate → defensive quality
        - key_pass_rate → creativity
        """
        return {
            # Ball Progression
            "progressive_pass_pct": self.value_to_percentile(
                stats.progressive_pass_pct, self.DISTRIBUTIONS["progressive_pass_pct"]
            ),
            "progressive_carry_pct": self.value_to_percentile(
                stats.progressive_carry_pct, self.DISTRIBUTIONS["progressive_carry_pct"]
            ),
            "final_third_pass_pct": self.value_to_percentile(
                stats.final_third_pass_pct, self.DISTRIBUTIONS["final_third_pass_pct"]
            ),
            "avg_pass_distance": 50.0,  # Default - requires tracking data
            # Defensive Contribution
            "pressing_rate": self.value_to_percentile(
                stats.pressures_per_90, self.DISTRIBUTIONS["pressures_per_90"]
            ),
            "tackle_success_rate": self.value_to_percentile(
                stats.tackle_success_rate, self.DISTRIBUTIONS["tackle_success_rate"]
            ),
            "interception_rate": self.value_to_percentile(
                stats.interceptions_per_90, self.DISTRIBUTIONS["interceptions_per_90"]
            ),
            "ball_recovery_rate": self.value_to_percentile(
                stats.ball_recoveries_per_90, self.DISTRIBUTIONS["ball_recoveries_per_90"]
            ),
            # Creativity
            "key_pass_rate": self.value_to_percentile(
                stats.key_pass_rate, self.DISTRIBUTIONS["key_pass_rate"]
            ),
            "through_ball_pct": self.value_to_percentile(
                stats.through_ball_pct, self.DISTRIBUTIONS["through_ball_pct"]
            ),
            "danger_creation_rate": 50.0,  # Default - requires tracking data
            # Work Rate & Positioning
            "central_presence_pct": 60.0,  # Default for CM
            "attacking_third_pct": 40.0,  # Default for CM
            "avg_speed": 50.0,  # Default - requires tracking data
            # Extra
            "pass_accuracy": self.value_to_percentile(
                stats.pass_accuracy, self.DISTRIBUTIONS["pass_accuracy"]
            ),
        }

    def compute_weights(self, stats: MidfielderStats) -> dict[str, float]:
        """Compute feature weights based on midfielder style.

        Weights adjusted based on playing patterns:
        - High pressure → more defensive weight
        - Progressive passing → more progression weight
        - Key passes → more creativity weight
        """
        weights = MIDFIELDER_WEIGHTS.copy()

        # Progressive passer: boost ball progression weights
        if stats.progressive_pass_pct > 20:
            weights["progressive_pass_pct"] += 0.08
            weights["progressive_carry_pct"] += 0.04

        # High presser: boost defensive weights
        if stats.pressures_per_90 > 25:
            weights["pressing_rate"] += 0.08
            weights["tackle_success_rate"] += 0.04

        # Creative player: boost creativity weights
        if stats.key_pass_rate > 3.0:
            weights["key_pass_rate"] += 0.08
            weights["through_ball_pct"] += 0.04

        # Ball carrier: boost carry weight
        if stats.progressive_carry_pct > 30:
            weights["progressive_carry_pct"] += 0.06

        # Normalize to sum to 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _compute_style_string(self, stats: MidfielderStats) -> str:
        """Determine midfielder style from stats."""
        notes = []

        # Primary role indicators
        if stats.pressures_per_90 > 25 and stats.tackle_success_rate > 55:
            notes.append("defensive midfielder")
        elif stats.key_pass_rate > 2.5 and stats.progressive_pass_pct > 18:
            notes.append("creative playmaker")
        elif stats.progressive_carry_pct > 28:
            notes.append("ball-carrying")

        # Secondary traits
        if stats.pressures_per_90 > 22:
            notes.append("high pressing")
        elif stats.pressures_per_90 < 12:
            notes.append("positional")

        if stats.progressive_pass_pct > 20:
            notes.append("progressive passer")

        if stats.pass_accuracy > 88:
            notes.append("high retention")
        elif stats.pass_accuracy < 78:
            notes.append("risk-taking")

        if stats.final_third_pass_pct > 30:
            notes.append("final third focus")

        if stats.through_ball_pct > 2.0:
            notes.append("through ball specialist")

        return ", ".join(notes) if notes else "balanced midfielder"

    def _get_summary_line(self, stats: MidfielderStats) -> str:
        """Get summary line with match/event count."""
        event_count = stats.passes + stats.pressures + stats.carries
        return f"Computed from {stats.matches} matches ({event_count} events)."
