"""Player archetype definition for similarity matching."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Archetype:
    """Define a player archetype for similarity matching."""

    name: str
    description: str = ""
    target_profile: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    directions: dict[str, int] = field(default_factory=dict)

    def set_target(self, feature: str, value: float) -> Archetype:
        self.target_profile[feature] = value
        return self

    def set_weight(self, feature: str, weight: float) -> Archetype:
        self.weights[feature] = weight
        return self

    def set_direction(self, feature: str, direction: int) -> Archetype:
        self.directions[feature] = direction
        return self

    def set_feature(self, feature: str, target: float, weight: float, direction: int = 1) -> Archetype:
        self.target_profile[feature] = target
        self.weights[feature] = weight
        self.directions[feature] = direction
        return self

    def get_features(self) -> list[str]:
        return [f for f, w in self.weights.items() if w > 0]

    def validate(self) -> list[str]:
        warnings = []
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.1:
            warnings.append(f"Weights sum to {weight_sum:.2f}, expected ~1.0")
        return warnings

    @classmethod
    def alvarez(cls) -> Archetype:
        """Create pre-built Julian Alvarez archetype."""
        archetype = cls(
            name="alvarez",
            description="Julian Alvarez: intelligent movement, spatial awareness, clinical finishing.",
        )
        archetype.target_profile = {
            "avg_separation": 85, "danger_rate": 90, "central_pct": 75,
            "avg_entry_speed": 70, "half_space_pct": 60, "avg_passing_options": 65, "carry_pct": 40,
        }
        archetype.weights = {
            "avg_separation": 0.23, "avg_entry_speed": 0.17, "avg_defensive_line_dist": 0.15,
            "central_pct": 0.12, "danger_rate": 0.18, "quick_break_pct": 0.05,
            "avg_teammates_ahead": 0.05, "half_space_pct": 0.02, "avg_passing_options": 0.02,
            "carry_pct": 0.00, "avg_distance": 0.01, "goal_rate": 0.00,
        }
        archetype.directions = {
            "avg_entry_speed": 1, "avg_distance": 1, "total_entries": 0, "avg_separation": 1,
            "avg_defensive_line_dist": -1, "central_pct": 1, "half_space_pct": 1, "carry_pct": 1,
            "avg_passing_options": 1, "avg_teammates_ahead": 1, "danger_rate": 1, "goal_rate": 1, "quick_break_pct": 1,
        }
        return archetype

    @classmethod
    def custom(cls, name: str, description: str = "") -> Archetype:
        return cls(name=name, description=description)
