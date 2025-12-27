"""Radar chart visualizations for player profile comparison.

This module creates radar charts to visually compare player profiles
against the Alvarez archetype and each other.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing import Sequence

# Features to display on radar chart (readable labels)
# * = target value is guessed (no StatsBomb tracking equivalent)
RADAR_FEATURES = {
    # Forward features
    "avg_separation": "Separation",
    "central_pct": "Central %*",
    "half_space_pct": "Half-Space %*",
    "danger_rate": "Danger Rate",
    "avg_entry_speed": "Speed*",
    "avg_passing_options": "Passing Options",
    "carry_pct": "Carry %",
    "avg_defensive_line_dist": "Depth*",
    # Defender features
    "stop_danger_rate": "Stop Danger %",
    "reduce_danger_rate": "Reduce Danger %",
    "force_backward_rate": "Force Back %",
    "pressing_rate": "Pressing %",
    "goal_side_rate": "Goal Side %",
    "avg_engagement_distance": "Engagement Dist",
    "beaten_by_possession_rate": "Beaten (Ball) %",
    "beaten_by_movement_rate": "Beaten (Move) %",
    # Goalkeeper features
    "pass_success_rate": "Pass Success %",
    "avg_pass_distance": "Pass Distance",
    "long_pass_pct": "Long Pass %",
    "short_pass_pct": "Short Pass %",
    "high_pass_pct": "High Pass %",
    "quick_distribution_pct": "Quick Dist %",
    "under_pressure_pct": "Under Pressure %",
    "to_attacking_third_pct": "To Attack 3rd %",
}


def normalize_for_radar(profiles: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """
    Normalize features to 0-100 scale for radar chart display.

    Uses min-max normalization within the dataset.
    """
    normalized = profiles.clone()

    for feature in features:
        if feature not in profiles.columns:
            continue

        min_val = profiles[feature].min()
        max_val = profiles[feature].max()

        # Avoid division by zero
        if max_val == min_val:
            normalized = normalized.with_columns(
                pl.lit(50.0).alias(f"{feature}_norm")
            )
        else:
            normalized = normalized.with_columns(
                ((pl.col(feature) - min_val) / (max_val - min_val) * 100)
                .alias(f"{feature}_norm")
            )

    return normalized


def normalize_for_radar_percentile(
    profiles: pl.DataFrame,
    all_profiles: pl.DataFrame,
    features: list[str]
) -> pl.DataFrame:
    """
    Normalize features to percentile ranks against full dataset.

    This provides more meaningful values than min-max within a small subset.
    A player at the 60th percentile shows as 60, making it directly comparable
    to target profile values which also represent percentiles.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Players to normalize (e.g., top 3)
    all_profiles : pl.DataFrame
        Full dataset to compute percentiles against
    features : list[str]
        Features to normalize

    Returns:
    --------
    pl.DataFrame with {feature}_norm columns containing percentile ranks (0-100)
    """
    normalized = profiles.clone()

    for feature in features:
        if feature not in profiles.columns or feature not in all_profiles.columns:
            continue

        # Get all values from full dataset (exclude nulls)
        all_values = all_profiles[feature].drop_nulls().to_list()

        if not all_values:
            normalized = normalized.with_columns(
                pl.lit(50.0).alias(f"{feature}_norm")
            )
            continue

        # Calculate percentile rank for each player in profiles
        percentile_ranks = []
        for val in profiles[feature].to_list():
            if val is None:
                percentile_ranks.append(50.0)
            else:
                # Percentile rank: % of values <= this value
                rank = sum(1 for v in all_values if v <= val) / len(all_values) * 100
                percentile_ranks.append(rank)

        normalized = normalized.with_columns(
            pl.Series(f"{feature}_norm", percentile_ranks)
        )

    return normalized


def get_alvarez_radar_profile(features: list[str]) -> dict:
    """
    Get Alvarez target profile formatted for radar chart.

    Returns normalised 0-100 values for each feature.
    """
    from src.utils.alvarez_profile import get_alvarez_target

    target = get_alvarez_target()
    return {"name": "Alvarez (Target)", **{f: target.get(f, 50) for f in features}}


def plot_radar_comparison(
    players: list[dict],
    features: list[str] | None = None,
    title: str = "Player Profile Comparison",
    figsize: tuple[int, int] = (10, 8),
    include_alvarez: bool = True,
    target_profile: dict | None = None,
    target_name: str = "Target",
) -> plt.Figure:
    """
    Create radar chart comparing multiple player profiles.

    Parameters:
    -----------
    players : list[dict]
        List of player profile dicts with feature values
        Each dict should have 'name' and feature values
    features : list[str], optional
        Features to include. Uses RADAR_FEATURES keys if not specified.
    title : str
        Chart title
    figsize : tuple
        Figure size
    include_alvarez : bool
        Whether to include target profile as reference (legacy name kept for compatibility)
    target_profile : dict, optional
        Custom target profile values. If None, uses Alvarez profile for forwards.
    target_name : str
        Name to display for the target profile line

    Returns:
    --------
    matplotlib Figure
    """
    if features is None:
        features = list(RADAR_FEATURES.keys())

    # Get labels for radar
    labels = [RADAR_FEATURES.get(f, f) for f in features]
    num_vars = len(features)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Add target profile first (so it appears behind)
    if include_alvarez:
        if target_profile is not None:
            target = {"name": f"{target_name} (Target)", **{f: target_profile.get(f, 50) for f in features}}
        else:
            target = get_alvarez_radar_profile(features)
        target_values = [target.get(f, 50) for f in features]
        target_values += target_values[:1]
        ax.plot(angles, target_values, '--', linewidth=2, label=target.get('name', 'Target'),
                color='#f39c12', alpha=0.8)
        ax.fill(angles, target_values, alpha=0.1, color='#f39c12')

    # Colors for different players
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']

    for i, player in enumerate(players):
        # Get values for each feature
        values = []
        for feature in features:
            val = player.get(feature, 0)
            if val is None:
                val = 0
            values.append(val)

        values += values[:1]  # Complete the loop

        # Plot
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=player.get('name', f'Player {i+1}'), color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)

    # Set y-axis range
    ax.set_ylim(0, 100)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Title
    plt.title(title, size=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    return fig


def plot_single_profile(
    player: dict,
    features: list[str] | None = None,
    title: str | None = None,
    color: str = '#3498db',
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Create radar chart for a single player profile.

    Parameters:
    -----------
    player : dict
        Player profile dict with feature values and 'name' key
    features : list[str], optional
        Features to include
    title : str, optional
        Chart title. Uses player name if not specified.
    color : str
        Fill color for the radar
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib Figure
    """
    if features is None:
        features = list(RADAR_FEATURES.keys())

    if title is None:
        title = f"{player.get('name', 'Player')} Profile"

    return plot_radar_comparison([player], features=features, title=title, figsize=figsize)


def plot_similarity_ranking(
    profiles: pl.DataFrame,
    top_n: int = 10,
    title: str = "Player Similarity Rankings",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create horizontal bar chart of player similarity rankings.

    Parameters:
    -----------
    profiles : pl.DataFrame
        Player profiles with similarity_score column
    top_n : int
        Number of top players to show
    title : str
        Chart title
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib Figure
    """
    top_players = profiles.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    names = top_players["player_name"].to_list()
    scores = top_players["similarity_score"].to_list()

    # Add team names if available
    if "team_name" in top_players.columns:
        teams = top_players["team_name"].to_list()
        names = [f"{n} ({t})" for n, t in zip(names, teams)]

    # Reverse for horizontal bar chart (top at top)
    names = names[::-1]
    scores = scores[::-1]

    # Create bars
    colors = plt.cm.RdYlGn(np.array(scores) / 100)
    bars = ax.barh(names, scores, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10)

    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_comparison_table(
    profiles: pl.DataFrame,
    features: list[str] | None = None,
    top_n: int = 5
) -> pl.DataFrame:
    """
    Create a formatted comparison table for top candidates.

    Returns a DataFrame suitable for display.
    """
    if features is None:
        features = list(RADAR_FEATURES.keys())

    # Select relevant columns
    display_cols = ["rank", "player_name"]
    if "team_name" in profiles.columns:
        display_cols.append("team_name")

    display_cols.extend(["similarity_score", "total_entries"])
    display_cols.extend([f for f in features if f in profiles.columns])

    available_cols = [c for c in display_cols if c in profiles.columns]

    return profiles.head(top_n).select(available_cols)
