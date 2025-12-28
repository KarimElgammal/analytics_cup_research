"""Forward mapper - StatsBomb to SkillCorner conversion."""

from __future__ import annotations

from src.archetypes.base import FORWARD_FEATURE_NAMES
from src.archetypes.forwards import FORWARD_COMPUTED, FORWARD_WEIGHTS
from src.statsbomb.mappers.base import BaseMapper
from src.statsbomb.stats import PlayerStats


class ForwardMapper(BaseMapper):
    """Map forward/attacker stats to SkillCorner target profiles."""

    DISTRIBUTIONS = {
        "conversion_rate": {"p10": 5.0, "p25": 8.0, "p50": 12.0, "p75": 18.0, "p90": 25.0},
        "shot_accuracy": {"p10": 25.0, "p25": 35.0, "p50": 45.0, "p75": 55.0, "p90": 65.0},
        "box_touches_per_90": {"p10": 2.0, "p25": 4.0, "p50": 7.0, "p75": 12.0, "p90": 18.0},
        "pass_accuracy": {"p10": 65.0, "p25": 72.0, "p50": 80.0, "p75": 86.0, "p90": 92.0},
        "dribble_success": {"p10": 35.0, "p25": 45.0, "p50": 55.0, "p75": 65.0, "p90": 75.0},
        "key_passes_per_90": {"p10": 0.3, "p25": 0.6, "p50": 1.0, "p75": 1.8, "p90": 2.5},
    }

    COMPUTED_FEATURES = FORWARD_COMPUTED
    FEATURE_NAMES = FORWARD_FEATURE_NAMES

    def map_to_target(self, stats: PlayerStats) -> dict[str, float]:
        """Map forward stats to SkillCorner target profile.

        Mapping logic:
        - conversion_rate → danger_rate (clinical finishing = dangerous entries)
        - box_touches_per_90 → avg_separation (finds space in box = good separation)
        - pass_accuracy → avg_passing_options (link-up play)
        """
        return {
            # Mapped from StatsBomb data
            "danger_rate": self.value_to_percentile(
                stats.conversion_rate, self.DISTRIBUTIONS["conversion_rate"]
            ),
            "avg_separation": self.value_to_percentile(
                stats.box_touches_per_90, self.DISTRIBUTIONS["box_touches_per_90"]
            ),
            "avg_passing_options": self.value_to_percentile(
                stats.pass_accuracy, self.DISTRIBUTIONS["pass_accuracy"]
            ),
            # Fixed values for tracking-specific metrics (no StatsBomb equivalent)
            "central_pct": 70.0,
            "avg_entry_speed": 65.0,
            "half_space_pct": 55.0,
            "quick_break_pct": 50.0,
            "avg_defensive_line_dist": 50.0,
            "avg_teammates_ahead": 40.0,
            "avg_distance": 50.0,
            "goal_rate": 50.0,
        }

    def compute_weights(self, stats: PlayerStats) -> dict[str, float]:
        """Compute feature weights from player statistics.

        Weights are adjusted based on the player's actual play style:
        - High conversion rate → higher danger_rate weight
        - High box touches → higher avg_separation weight
        - High key passes → higher avg_passing_options weight
        - High dribble success → higher avg_entry_speed weight
        """
        # Get percentile scores for adjustment
        dribble_score = (
            self.value_to_percentile(
                stats.dribble_success, self.DISTRIBUTIONS["dribble_success"]
            )
            / 100.0
        )
        conversion_score = (
            self.value_to_percentile(
                stats.conversion_rate, self.DISTRIBUTIONS["conversion_rate"]
            )
            / 100.0
        )
        box_touch_score = (
            self.value_to_percentile(
                stats.box_touches_per_90, self.DISTRIBUTIONS["box_touches_per_90"]
            )
            / 100.0
        )
        key_pass_score = (
            self.value_to_percentile(
                stats.key_passes_per_90, self.DISTRIBUTIONS["key_passes_per_90"]
            )
            / 100.0
        )

        # Start with base weights
        weights = FORWARD_WEIGHTS.copy()

        # Dribblers: boost entry speed
        if dribble_score > 0.5:
            boost = (dribble_score - 0.5) * 0.4
            weights["avg_entry_speed"] += boost * 0.5

        # Clinical finishers: boost danger_rate
        if conversion_score > 0.6:
            weights["danger_rate"] += (conversion_score - 0.6) * 0.3

        # Movement players: boost separation
        if box_touch_score > 0.7:
            weights["avg_separation"] += (box_touch_score - 0.7) * 0.3

        # Creative players: boost passing options
        if key_pass_score > 0.5:
            weights["avg_passing_options"] += (key_pass_score - 0.5) * 0.3

        # Normalize to sum to 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _compute_style_string(self, stats: PlayerStats) -> str:
        """Determine playing style from stats."""
        notes = []

        if stats.conversion_rate > 18:
            notes.append("clinical finisher")
        elif stats.conversion_rate < 10:
            notes.append("volume shooter")

        if stats.dribble_success > 60:
            notes.append("skilled dribbler")
        elif stats.dribble_success < 45:
            notes.append("movement-focused")

        if stats.box_touches_per_90 > 10:
            notes.append("box presence")

        if stats.key_passes_per_90 > 1.5:
            notes.append("creative")

        return ", ".join(notes) if notes else "balanced forward"

    def _get_summary_line(self, stats: PlayerStats) -> str:
        """Get summary line with match/minute count."""
        return f"Computed from {stats.matches} matches (~{stats.minutes:.0f} minutes)."
