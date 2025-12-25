"""Similarity scoring for player profile matching.

This module computes weighted cosine similarity between A-League player
profiles and the Alvarez archetype target profile.
"""

import numpy as np
import polars as pl
from src.utils.alvarez_profile import (
    FEATURE_WEIGHTS,
    FEATURE_DIRECTION,
    get_weighted_features,
)


def standardize_features(profiles: pl.DataFrame, features: list[str]) -> tuple[pl.DataFrame, dict]:
    """
    Standardize features using z-score normalization.

    Returns the standardized DataFrame and a dict of (mean, std) for each feature.
    """
    stats = {}
    standardized = profiles.clone()

    for feature in features:
        if feature not in profiles.columns:
            continue

        mean_val = profiles[feature].mean()
        std_val = profiles[feature].std()

        # Avoid division by zero
        if std_val is None or std_val == 0:
            std_val = 1.0
        if mean_val is None:
            mean_val = 0.0

        stats[feature] = {"mean": mean_val, "std": std_val}
        standardized = standardized.with_columns(
            ((pl.col(feature) - mean_val) / std_val).alias(feature)
        )

    return standardized, stats


def compute_target_profile(profiles: pl.DataFrame, features: list[str]) -> dict:
    """
    Compute the Alvarez target profile.

    For features where higher is better (direction=1), the target is the max.
    For features where lower is better (direction=-1), the target is the min.

    This creates an "ideal" Alvarez-like profile based on the best values
    observed in the data.
    """
    target = {}

    for feature in features:
        if feature not in profiles.columns:
            continue

        direction = FEATURE_DIRECTION.get(feature, 1)

        if direction == 1:
            # Higher is better - use 90th percentile to avoid outliers
            target[feature] = profiles[feature].quantile(0.90)
        elif direction == -1:
            # Lower is better - use 10th percentile
            target[feature] = profiles[feature].quantile(0.10)
        else:
            # Neutral - use median
            target[feature] = profiles[feature].median()

    return target


def compute_similarity_scores(
    profiles: pl.DataFrame,
    target: dict | None = None,
    weights: dict | None = None
) -> pl.DataFrame:
    """
    Compute weighted similarity scores between player profiles and target.

    Uses weighted cosine similarity on standardized features.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Player profiles with numeric features
    target : dict, optional
        Target profile values. If None, computed from data using ideal values.
    weights : dict, optional
        Feature weights. If None, uses FEATURE_WEIGHTS from alvarez_profile.

    Returns:
    --------
    pl.DataFrame with player info and similarity_score column
    """
    if weights is None:
        weights = FEATURE_WEIGHTS

    features = [f for f in get_weighted_features() if f in profiles.columns]

    if not features:
        raise ValueError("No valid features found for similarity computation")

    # Standardize features
    standardized, stats = standardize_features(profiles, features)

    # Compute target profile if not provided
    if target is None:
        target = compute_target_profile(profiles, features)

    # Standardize target using same stats
    target_std = {}
    for feature in features:
        if feature in target and feature in stats:
            target_std[feature] = (target[feature] - stats[feature]["mean"]) / stats[feature]["std"]
        else:
            target_std[feature] = 0.0

    # Compute weighted cosine similarity for each player
    def compute_cosine_similarity(row: dict) -> float:
        """Compute weighted cosine similarity between row and target."""
        player_vec = []
        target_vec = []
        weight_vec = []

        for feature in features:
            if feature in row and row[feature] is not None:
                player_val = row[feature]
                target_val = target_std.get(feature, 0.0)
                weight = weights.get(feature, 0.0)

                # Handle NaN values
                if np.isnan(player_val):
                    continue

                player_vec.append(player_val)
                target_vec.append(target_val)
                weight_vec.append(weight)

        if not player_vec:
            return 0.0

        player_arr = np.array(player_vec)
        target_arr = np.array(target_vec)
        weight_arr = np.array(weight_vec)

        # Weighted vectors
        player_weighted = player_arr * weight_arr
        target_weighted = target_arr * weight_arr

        # Cosine similarity
        dot_product = np.dot(player_weighted, target_weighted)
        player_norm = np.linalg.norm(player_weighted)
        target_norm = np.linalg.norm(target_weighted)

        if player_norm == 0 or target_norm == 0:
            return 0.0

        similarity = dot_product / (player_norm * target_norm)

        # Scale to 0-100
        return max(0, min(100, (similarity + 1) * 50))

    # Apply similarity computation
    rows = standardized.to_dicts()
    scores = [compute_cosine_similarity(row) for row in rows]

    # Add scores to original (non-standardized) profiles
    return profiles.with_columns(
        pl.Series("similarity_score", scores).round(1)
    ).sort("similarity_score", descending=True)


def rank_candidates(
    profiles: pl.DataFrame,
    min_entries: int = 3,
    top_n: int | None = None
) -> pl.DataFrame:
    """
    Rank player candidates by similarity to Alvarez archetype.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Player profiles (already with similarity_score column)
    min_entries : int
        Minimum entries required for inclusion
    top_n : int, optional
        Return only top N candidates

    Returns:
    --------
    Ranked DataFrame with similarity scores and rank column
    """
    # Filter by minimum entries
    ranked = profiles.filter(pl.col("total_entries") >= min_entries)

    # Sort by similarity score
    ranked = ranked.sort("similarity_score", descending=True)

    # Add rank column
    ranked = ranked.with_row_index("rank", offset=1)

    if top_n is not None:
        ranked = ranked.head(top_n)

    return ranked


def get_similarity_breakdown(
    player_profile: dict,
    features: list[str],
    weights: dict | None = None
) -> list[dict]:
    """
    Get detailed breakdown of similarity score by feature.

    Useful for explaining why a player is similar/different to target.
    """
    if weights is None:
        weights = FEATURE_WEIGHTS

    breakdown = []
    for feature in features:
        if feature in player_profile:
            breakdown.append({
                "feature": feature,
                "value": player_profile.get(feature),
                "weight": weights.get(feature, 0),
                "weighted_contribution": (
                    player_profile.get(feature, 0) * weights.get(feature, 0)
                ),
            })

    return sorted(breakdown, key=lambda x: x["weight"], reverse=True)
