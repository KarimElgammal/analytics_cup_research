"""Base mapper class for StatsBomb to SkillCorner conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np

# Generic type for stats dataclasses
StatsT = TypeVar("StatsT")


class BaseMapper(ABC):
    """Abstract base class for position-specific mappers.

    Subclasses must define:
    - DISTRIBUTIONS: Reference percentile distributions
    - COMPUTED_FEATURES: Features computed from data
    - FEATURE_NAMES: Display names for features
    """

    DISTRIBUTIONS: dict[str, dict[str, float]]
    COMPUTED_FEATURES: frozenset[str]
    FEATURE_NAMES: dict[str, str]

    @classmethod
    def value_to_percentile(cls, value: float, distribution: dict[str, float]) -> float:
        """Map a raw value to 0-100 percentile using reference distribution.

        Uses linear interpolation between percentile thresholds for smooth mapping.
        Optimized with early returns and minimal branching.

        Args:
            value: Raw statistic value
            distribution: Dict with p10, p25, p50, p75, p90 thresholds

        Returns:
            Percentile value between 0 and 100
        """
        if value <= 0:
            return 0.0

        p10, p25, p50, p75, p90 = (
            distribution["p10"],
            distribution["p25"],
            distribution["p50"],
            distribution["p75"],
            distribution["p90"],
        )

        # Use numpy for vectorized comparison (faster for batches)
        if value <= p10:
            return 10 * (value / p10) if p10 > 0 else 0
        if value <= p25:
            return 10 + 15 * ((value - p10) / (p25 - p10)) if (p25 - p10) > 0 else 10
        if value <= p50:
            return 25 + 25 * ((value - p25) / (p50 - p25)) if (p50 - p25) > 0 else 25
        if value <= p75:
            return 50 + 25 * ((value - p50) / (p75 - p50)) if (p75 - p50) > 0 else 50
        if value <= p90:
            return 75 + 15 * ((value - p75) / (p90 - p75)) if (p90 - p75) > 0 else 75

        # Above p90: extrapolate but cap at 100
        extra = 10 * ((value - p90) / p90) if p90 > 0 else 0
        return min(100.0, 90 + extra)

    @classmethod
    def batch_to_percentile(
        cls,
        values: np.ndarray,
        distribution: dict[str, float],
    ) -> np.ndarray:
        """Vectorized percentile conversion for arrays.

        Much faster than calling value_to_percentile in a loop.

        Args:
            values: Array of raw values
            distribution: Dict with p10, p25, p50, p75, p90 thresholds

        Returns:
            Array of percentile values (0-100)
        """
        p10, p25, p50, p75, p90 = (
            distribution["p10"],
            distribution["p25"],
            distribution["p50"],
            distribution["p75"],
            distribution["p90"],
        )

        result = np.zeros_like(values, dtype=np.float64)

        # Handle each percentile band
        mask = values <= 0
        result[mask] = 0

        mask = (values > 0) & (values <= p10)
        if p10 > 0:
            result[mask] = 10 * (values[mask] / p10)

        mask = (values > p10) & (values <= p25)
        if (p25 - p10) > 0:
            result[mask] = 10 + 15 * ((values[mask] - p10) / (p25 - p10))

        mask = (values > p25) & (values <= p50)
        if (p50 - p25) > 0:
            result[mask] = 25 + 25 * ((values[mask] - p25) / (p50 - p25))

        mask = (values > p50) & (values <= p75)
        if (p75 - p50) > 0:
            result[mask] = 50 + 25 * ((values[mask] - p50) / (p75 - p50))

        mask = (values > p75) & (values <= p90)
        if (p90 - p75) > 0:
            result[mask] = 75 + 15 * ((values[mask] - p75) / (p90 - p75))

        mask = values > p90
        if p90 > 0:
            result[mask] = np.minimum(100.0, 90 + 10 * ((values[mask] - p90) / p90))

        return result

    @abstractmethod
    def map_to_target(self, stats: StatsT) -> dict[str, float]:
        """Map position-specific stats to SkillCorner target profile.

        Args:
            stats: Position-specific stats dataclass

        Returns:
            Dictionary with target profile values (0-100 scale)
        """
        ...

    @abstractmethod
    def compute_weights(self, stats: StatsT) -> dict[str, float]:
        """Compute dynamic feature weights based on player style.

        Args:
            stats: Position-specific stats dataclass

        Returns:
            Dictionary with feature weights summing to 1.0
        """
        ...

    def get_description(self, stats: StatsT, target: dict[str, float]) -> str:
        """Generate human-readable archetype description.

        Default implementation - subclasses can override for custom formatting.

        Args:
            stats: Position-specific stats dataclass
            target: Mapped target profile

        Returns:
            Multi-line description string with style info and target table
        """
        style_str = self._compute_style_string(stats)
        summary = self._get_summary_line(stats)

        table_lines = [
            "Target profile (percentile targets 0-100):",
            "| Feature | Target | Source |",
            "|---------|--------|--------|",
        ]

        for key, value in target.items():
            display = self.FEATURE_NAMES.get(key, key.replace("_", " ").title())
            source = "StatsBomb" if key in self.COMPUTED_FEATURES else "Estimated"
            table_lines.append(f"| {display} | {value:.0f} | {source} |")

        return f"{summary} {style_str.capitalize()}.\n\n" + "\n".join(table_lines)

    @abstractmethod
    def _compute_style_string(self, stats: StatsT) -> str:
        """Compute style description from stats."""
        ...

    @abstractmethod
    def _get_summary_line(self, stats: StatsT) -> str:
        """Get summary line (e.g., 'Computed from X matches (Y events).')."""
        ...
