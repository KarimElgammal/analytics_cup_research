"""Base archetype configuration classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.archetype import Archetype


class Position(Enum):
    """Player position types."""
    FORWARD = auto()
    DEFENDER = auto()
    GOALKEEPER = auto()
    MIDFIELDER = auto()


@dataclass(frozen=True, slots=True)
class ArchetypeConfig:
    """Immutable archetype configuration.

    Using frozen=True for hashability and slots=True for memory efficiency.

    Attributes:
        key: Unique identifier (e.g., "alvarez", "gvardiol")
        name: Display name (e.g., "Alvarez (ARG) - Movement-focused")
        description: Brief description of playing style
        target_profile: Dict of feature -> percentile target (0-100)
        position: Position type for this archetype
        computed_features: Set of features computed from data (vs estimated)
        source_matches: Number of matches used to compute profile
        source_events: Number of events analyzed
    """
    key: str
    name: str
    description: str
    target_profile: dict[str, float]
    position: Position
    computed_features: frozenset[str] = field(default_factory=frozenset)
    source_matches: int = 0
    source_events: int = 0

    def to_archetype(
        self,
        weights: dict[str, float],
        directions: dict[str, int],
        feature_names: dict[str, str] | None = None,
    ) -> Archetype:
        """Convert config to Archetype instance.

        Args:
            weights: Feature weights for similarity scoring
            directions: Feature directions (1=higher is better, -1=lower is better)
            feature_names: Optional display names for features

        Returns:
            Archetype instance ready for similarity matching
        """
        from src.core.archetype import Archetype

        full_desc = self._format_description(feature_names or {})
        return Archetype(
            name=self.key,
            description=full_desc,
            target_profile=dict(self.target_profile),
            weights=weights,
            directions=directions,
        )

    def _format_description(self, feature_names: dict[str, str]) -> str:
        """Format full description with target table."""
        # Header line
        if self.source_matches > 0:
            header = f"{self.name}\n\nComputed from {self.source_matches} matches"
            if self.source_events > 0:
                header += f" ({self.source_events} events)"
            header += f". {self.description}"
        else:
            header = f"{self.name}\n\n{self.description}"

        # Target table
        lines = [
            "\n\nTarget profile (percentile targets 0-100):",
            "| Feature | Target | Source |",
            "|---------|--------|--------|",
        ]

        for key, value in self.target_profile.items():
            display = feature_names.get(key, key.replace("_", " ").title())
            source = "StatsBomb" if key in self.computed_features else "Estimated"
            lines.append(f"| {display} | {value:.0f} | {source} |")

        return header + "\n".join(lines)


# Common feature name mappings
FORWARD_FEATURE_NAMES: dict[str, str] = {
    "danger_rate": "Danger Rate",
    "avg_separation": "Separation",
    "central_pct": "Central %",
    "avg_entry_speed": "Speed",
    "avg_passing_options": "Passing Options",
    "half_space_pct": "Half Space %",
    "quick_break_pct": "Quick Break %",
    "avg_defensive_line_dist": "Def Line Dist",
    "avg_teammates_ahead": "Teammates Ahead",
}

DEFENDER_FEATURE_NAMES: dict[str, str] = {
    "stop_danger_rate": "Stop Danger %",
    "reduce_danger_rate": "Reduce Danger %",
    "force_backward_rate": "Force Back %",
    "beaten_by_possession_rate": "Beaten (Ball) %",
    "beaten_by_movement_rate": "Beaten (Move) %",
    "pressing_rate": "Pressing %",
    "goal_side_rate": "Goal Side %",
    "avg_engagement_distance": "Engagement Dist",
    "defensive_third_pct": "Defensive 3rd %",
    "middle_third_pct": "Middle 3rd %",
    "attacking_third_pct": "Attacking 3rd %",
}

GOALKEEPER_FEATURE_NAMES: dict[str, str] = {
    "pass_success_rate": "Pass Success %",
    "short_pass_pct": "Short Pass %",
    "long_pass_pct": "Long Pass %",
    "avg_pass_distance": "Pass Distance",
    "quick_distribution_pct": "Quick Dist %",
    "high_pass_pct": "High Pass %",
    "to_middle_third_pct": "To Middle 3rd %",
    "to_attacking_third_pct": "To Attack 3rd %",
    "avg_passing_options": "Passing Options",
}

MIDFIELDER_FEATURE_NAMES: dict[str, str] = {
    # Ball Progression
    "progressive_pass_pct": "Progressive Pass %",
    "progressive_carry_pct": "Progressive Carry %",
    "final_third_pass_pct": "Final Third Pass %",
    "avg_pass_distance": "Pass Distance",
    # Defensive
    "pressing_rate": "Pressing Rate",
    "tackle_success_rate": "Tackle Success %",
    "interception_rate": "Interception Rate",
    "ball_recovery_rate": "Ball Recovery Rate",
    # Creativity
    "key_pass_rate": "Key Pass Rate",
    "through_ball_pct": "Through Ball %",
    "danger_creation_rate": "Danger Creation %",
    # Work Rate
    "central_presence_pct": "Central Presence %",
    "attacking_third_pct": "Attacking Third %",
    "avg_speed": "Avg Speed",
    # Extra
    "pass_accuracy": "Pass Accuracy %",
}
