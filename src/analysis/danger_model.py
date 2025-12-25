"""XGBoost model for danger prediction and feature importance.

Uses ML to empirically determine which features best predict
dangerous entries, validating our Alvarez archetype weights.
"""

import polars as pl
import numpy as np
from typing import Tuple


def prepare_features(entries: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare features and labels for danger prediction.

    Returns:
        X: Feature matrix
        y: Binary labels (lead_to_shot)
        feature_names: List of feature names
    """
    feature_cols = [
        "speed_avg",
        "separation_end",
        "delta_to_last_defensive_line_end",
        "n_passing_options_ahead",
        "n_teammates_ahead_end",
        "n_opponents_ahead_end",
        "distance_covered",
        "x_end",
        "y_end",
    ]

    # Add categorical features as numeric
    # Entry zone: central=2, half_space=1, wide=0
    if "entry_zone" in entries.columns:
        entries = entries.with_columns(
            pl.when(pl.col("entry_zone") == "central").then(2)
            .when(pl.col("entry_zone") == "half_space").then(1)
            .otherwise(0)
            .alias("entry_zone_num")
        )
        feature_cols.append("entry_zone_num")

    # Entry method: carry=1, pass=0
    if "entry_method" in entries.columns:
        entries = entries.with_columns(
            pl.when(pl.col("entry_method") == "carry").then(1)
            .otherwise(0)
            .alias("entry_method_num")
        )
        feature_cols.append("entry_method_num")

    # Phase: quick_break=2, transition=1, other=0
    if "team_in_possession_phase_type" in entries.columns:
        entries = entries.with_columns(
            pl.when(pl.col("team_in_possession_phase_type") == "quick_break").then(2)
            .when(pl.col("team_in_possession_phase_type") == "transition").then(1)
            .otherwise(0)
            .alias("phase_num")
        )
        feature_cols.append("phase_num")

    # Filter to available columns
    available_cols = [c for c in feature_cols if c in entries.columns]

    # Get labels
    if "is_dangerous" in entries.columns:
        y = entries["is_dangerous"].to_numpy().astype(int)
    elif "lead_to_shot" in entries.columns:
        y = entries["lead_to_shot"].to_numpy().astype(int)
    else:
        raise ValueError("No danger label found")

    # Get features
    X = entries.select(available_cols).to_numpy()

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    return X, y, available_cols


def train_danger_model(entries: pl.DataFrame) -> dict:
    """
    Train gradient boosting model to predict dangerous entries.

    Tries LightGBM first, falls back to sklearn GradientBoosting.
    Returns dict with model, feature importances, and metrics.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier

    X, y, feature_names = prepare_features(entries)

    # Try LightGBM first (faster, handles imbalance well)
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        model_name = "LightGBM"
    except ImportError:
        # Fall back to sklearn
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model_name = "GradientBoosting"

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    # Fit on all data for feature importance
    model.fit(X, y)

    # Get feature importances
    importances = dict(zip(feature_names, model.feature_importances_))

    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    return {
        "model": model,
        "model_name": model_name,
        "feature_importances": importances,
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
        "feature_names": feature_names,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "positive_rate": y.mean(),
    }


def get_importance_based_weights(importances: dict) -> dict:
    """
    Convert XGBoost feature importances to similarity weights.

    Maps model features to profile features and normalizes.
    """
    # Mapping from model features to profile features
    feature_mapping = {
        "speed_avg": "avg_entry_speed",
        "separation_end": "avg_separation",
        "delta_to_last_defensive_line_end": "avg_defensive_line_dist",
        "n_passing_options_ahead": "avg_passing_options",
        "n_teammates_ahead_end": "avg_teammates_ahead",
        "entry_zone_num": "central_pct",  # Proxy
        "phase_num": "quick_break_pct",  # Proxy
    }

    weights = {}
    for model_feat, importance in importances.items():
        if model_feat in feature_mapping:
            profile_feat = feature_mapping[model_feat]
            weights[profile_feat] = importance

    # Add danger_rate (the target itself, so moderate weight)
    weights["danger_rate"] = 0.15

    # Normalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def print_importance_report(result: dict) -> str:
    """Generate a report of feature importances."""
    lines = [
        "# XGBoost Danger Prediction Model",
        "",
        f"**Samples**: {result['n_samples']} entries",
        f"**Positive rate**: {result['positive_rate']:.1%} lead to shots",
        f"**CV AUC**: {result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f}",
        "",
        "## Feature Importances",
        "",
        "| Feature | Importance |",
        "|---------|------------|",
    ]

    for feat, imp in result['feature_importances'].items():
        lines.append(f"| {feat} | {imp:.3f} |")

    return "\n".join(lines)
