"""Similarity scoring engine for player archetype matching."""

from __future__ import annotations
import numpy as np
import polars as pl
from src.core.archetype import Archetype


class SimilarityEngine:
    """Compute similarity between player profiles and an archetype."""

    def __init__(self, archetype: Archetype):
        self.archetype = archetype
        self.profiles: pl.DataFrame | None = None
        self.rankings: pl.DataFrame | None = None
        self._stats: dict[str, dict[str, float]] = {}

    def fit(self, profiles: pl.DataFrame) -> SimilarityEngine:
        self.profiles = profiles
        self._compute_stats()
        self._compute_scores()
        return self

    def _compute_stats(self) -> None:
        if self.profiles is None:
            return
        for feature in self._get_available_features():
            if feature not in self.profiles.columns:
                continue
            mean_val = self.profiles[feature].mean() or 0.0
            std_val = self.profiles[feature].std() or 1.0
            self._stats[feature] = {"mean": mean_val, "std": std_val if std_val else 1.0}

    def _get_available_features(self) -> list[str]:
        if self.profiles is None:
            return []
        return [f for f, w in self.archetype.weights.items() if w > 0 and f in self.profiles.columns]

    def _standardize_value(self, feature: str, value: float) -> float:
        if feature not in self._stats:
            return 0.0
        return (value - self._stats[feature]["mean"]) / self._stats[feature]["std"]

    def _compute_target_profile(self) -> dict[str, float]:
        target_std = {}
        for feature in self._get_available_features():
            if feature in self.archetype.target_profile and self.profiles is not None:
                pct = self.archetype.target_profile[feature] / 100.0
                actual_target = self.profiles[feature].quantile(pct)
                target_std[feature] = self._standardize_value(feature, actual_target)
            else:
                direction = self.archetype.directions.get(feature, 1)
                if self.profiles is not None and feature in self.profiles.columns:
                    target_val = self.profiles[feature].quantile(0.90 if direction == 1 else 0.10)
                    target_std[feature] = self._standardize_value(feature, target_val)
        return target_std

    def _compute_scores(self) -> None:
        if self.profiles is None:
            return
        features = self._get_available_features()
        target_std = self._compute_target_profile()
        weights = self.archetype.weights

        def compute_cosine_similarity(row: dict) -> float:
            player_vec, target_vec, weight_vec = [], [], []
            for feature in features:
                if feature not in row or row[feature] is None:
                    continue
                player_val = self._standardize_value(feature, row[feature])
                if np.isnan(player_val):
                    continue
                player_vec.append(player_val)
                target_vec.append(target_std.get(feature, 0.0))
                weight_vec.append(weights.get(feature, 0.0))
            if not player_vec:
                return 0.0
            player_arr, target_arr, weight_arr = np.array(player_vec), np.array(target_vec), np.array(weight_vec)
            dot_product = np.dot(player_arr * weight_arr, target_arr * weight_arr)
            player_norm, target_norm = np.linalg.norm(player_arr * weight_arr), np.linalg.norm(target_arr * weight_arr)
            if player_norm == 0 or target_norm == 0:
                return 0.0
            return max(0, min(100, (dot_product / (player_norm * target_norm) + 1) * 50))

        scores = [compute_cosine_similarity(row) for row in self.profiles.to_dicts()]
        self.profiles = self.profiles.with_columns(pl.Series("similarity_score", scores).round(1)).sort("similarity_score", descending=True)

    def score(self) -> pl.DataFrame:
        if self.profiles is None:
            raise ValueError("Engine not fitted. Call fit() first.")
        return self.profiles

    def rank(self, top_n: int = 10, min_entries: int | None = None) -> pl.DataFrame:
        if self.profiles is None:
            raise ValueError("Engine not fitted. Call fit() first.")
        ranked = self.profiles.sort("similarity_score", descending=True)
        if min_entries is not None:
            ranked = ranked.filter(pl.col("total_entries") >= min_entries)
        self.rankings = ranked.with_row_index("rank", offset=1).head(top_n)
        return self.rankings

    def explain(self, player_name: str) -> dict:
        if self.profiles is None:
            raise ValueError("Engine not fitted.")
        player = self.profiles.filter(pl.col("player_name") == player_name)
        if len(player) == 0:
            raise ValueError(f"Player '{player_name}' not found")
        return {"player_name": player_name, "similarity_score": player.to_dicts()[0].get("similarity_score", 0), "archetype": self.archetype.name}

    def get_feature_importance(self) -> list[dict]:
        return sorted([{"feature": f, "weight": self.archetype.weights.get(f, 0)} for f in self._get_available_features()], key=lambda x: x["weight"], reverse=True)
