"""AI-powered player similarity insights using GitHub Models or HuggingFace Inference.

Refactored with class-based architecture for cleaner code and richer prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for an AI model."""
    key: str
    model_id: str
    display_name: str
    backend: str  # "github" or "huggingface"
    max_tokens: int = 500
    temperature: float = 0.7


# Available models
MODELS: dict[str, ModelConfig] = {
    # GitHub Models
    "grok-3-mini": ModelConfig("grok-3-mini", "xai/grok-3-mini", "Grok 3 Mini (GitHub) - expensive", "github", max_tokens=2000),
    "phi-4": ModelConfig("phi-4", "microsoft/Phi-4", "Phi-4 (GitHub)", "github"),
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai/gpt-4o-mini", "GPT-4o Mini (GitHub)", "github"),
    # HuggingFace Models
    "llama-3.1-8b": ModelConfig("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B (HF)", "huggingface"),
    "llama-3.2-3b": ModelConfig("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B (HF)", "huggingface"),
    "qwen-2.5-7b": ModelConfig("qwen-2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B (HF)", "huggingface"),
    "smollm3-3b": ModelConfig("smollm3-3b", "HuggingFaceTB/SmolLM3-3B", "SmolLM3 3B (HF)", "huggingface"),
    "gemma-2-2b": ModelConfig("gemma-2-2b", "google/gemma-2-2b-it", "Gemma 2 2B (HF)", "huggingface"),
}

HF_MODELS = {k for k, v in MODELS.items() if v.backend == "huggingface"}
GITHUB_MODELS = {k for k, v in MODELS.items() if v.backend == "github"}

DEFAULT_HF_MODEL = "llama-3.1-8b"
DEFAULT_GITHUB_MODEL = "phi-4"

# For backwards compatibility
MODEL_DISPLAY_NAMES = {k: v.display_name for k, v in MODELS.items()}
AVAILABLE_MODELS = {k: v.model_id for k, v in MODELS.items()}


# =============================================================================
# Position Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class MetricInfo:
    """Information about a metric."""
    key: str
    label: str
    definition: str
    format_type: str = "percentage"  # "percentage", "distance", "speed", "number"


@dataclass(frozen=True, slots=True)
class PositionConfig:
    """Configuration for a position type."""
    name: str
    metrics: tuple[MetricInfo, ...]
    count_field: str
    criteria: str

    def get_metric_keys(self) -> list[str]:
        return [m.key for m in self.metrics]


# Metric definitions
FORWARD_METRICS = (
    MetricInfo("danger_rate", "Danger Rate", "Percentage of entries leading to shots. Higher = more clinical.", "percentage"),
    MetricInfo("central_pct", "Central %", "Percentage through central zone. Central = striker-like role.", "percentage"),
    MetricInfo("avg_separation", "Separation", "Distance from nearest defender (m). Higher = better movement.", "distance"),
    MetricInfo("avg_entry_speed", "Entry Speed", "Speed entering final third (m/s). Higher = direct runs.", "speed"),
    MetricInfo("avg_defensive_line_dist", "Depth", "Distance from defensive line (m). Lower = closer to goal.", "distance"),
    MetricInfo("quick_break_pct", "Quick Break %", "Entries during counter-attacks. High = transition threat.", "percentage"),
    MetricInfo("avg_passing_options", "Pass Options", "Teammates available for pass. Higher = good timing.", "number"),
    MetricInfo("half_space_pct", "Half-Space %", "Entries through half-spaces. Creative playmaker zones.", "percentage"),
    # SkillCorner advanced metrics
    MetricInfo("one_touch_pct", "One Touch %", "Entries involving one-touch play. Quick combination play.", "percentage"),
    MetricInfo("penalty_area_pct", "Penalty Area %", "Entries ending in penalty area. Central finishing positions.", "percentage"),
    MetricInfo("avg_opponents_bypassed", "Opponents Bypassed", "Average opponents bypassed per entry.", "number"),
    MetricInfo("forward_momentum_pct", "Forward Momentum %", "Entries with forward momentum. Attacking intent.", "percentage"),
    # Transition speed metrics (new)
    MetricInfo("avg_transition_speed", "Transition Speed", "Seconds from possession start to entry. Lower = faster.", "number"),
    MetricInfo("fast_transition_pct", "Fast Transition %", "Entries within 3 seconds of possession start.", "percentage"),
    # Passer/Receiver credit metrics (new)
    MetricInfo("assisted_pct", "Assisted %", "Entries received from teammate pass vs solo carries.", "percentage"),
    MetricInfo("assisted_danger_rate", "Assisted Danger %", "Danger rate on assisted entries. Finishing quality.", "percentage"),
    MetricInfo("solo_danger_rate", "Solo Danger %", "Danger rate on unassisted entries. Individual threat.", "percentage"),
    MetricInfo("total_entry_assists", "Entry Assists", "Passes enabling teammate entries. Playmaking ability.", "number"),
    MetricInfo("assist_danger_rate", "Assist Danger %", "Entry assists leading to shots. Creative impact.", "percentage"),
)

DEFENDER_METRICS = (
    MetricInfo("stop_danger_rate", "Stop Danger %", "Engagements completely stopping dangerous attacks.", "percentage"),
    MetricInfo("reduce_danger_rate", "Reduce Danger %", "Engagements reducing but not eliminating danger.", "percentage"),
    MetricInfo("pressing_rate", "Pressing %", "How often defender engages proactively.", "percentage"),
    MetricInfo("goal_side_rate", "Goal-Side %", "Engagements with goal-side position maintained.", "percentage"),
    MetricInfo("avg_engagement_distance", "Engagement Dist", "Distance from own goal when engaging (m).", "distance"),
    MetricInfo("force_backward_rate", "Force Back %", "Engagements forcing attacker to play backwards.", "percentage"),
    MetricInfo("beaten_by_movement_rate", "Beaten (Move) %", "How often defender is beaten by off-ball movement.", "percentage"),
    MetricInfo("beaten_by_possession_rate", "Beaten (Ball) %", "How often defender is beaten while opponent keeps ball.", "percentage"),
    # SkillCorner advanced metrics
    MetricInfo("avg_engagement_angle", "Engagement Angle", "Average angle of engagement (degrees).", "number"),
    MetricInfo("avg_consecutive_engagements", "Consecutive Engagements", "Average consecutive engagements. Sustained pressure.", "number"),
    MetricInfo("close_at_start_pct", "Close at Start %", "Engagements where defender was close at start.", "percentage"),
    MetricInfo("avg_possession_danger", "Possession Danger", "Average danger level of possessions faced.", "number"),
)

GOALKEEPER_METRICS = (
    MetricInfo("pass_success_rate", "Pass Success %", "Distributions reaching teammate successfully.", "percentage"),
    MetricInfo("avg_pass_distance", "Pass Distance", "Average distribution distance (m).", "distance"),
    MetricInfo("long_pass_pct", "Long Pass %", "Distributions that are long passes.", "percentage"),
    MetricInfo("quick_distribution_pct", "Quick Dist %", "Distributions made quickly.", "percentage"),
    MetricInfo("short_pass_pct", "Short Pass %", "Distributions that are short passes.", "percentage"),
    MetricInfo("high_pass_pct", "High Pass %", "Distributions that are aerial/lofted passes.", "percentage"),
    MetricInfo("to_attacking_third_pct", "To Attack 3rd %", "Distributions reaching the attacking third.", "percentage"),
    # SkillCorner advanced metrics
    MetricInfo("pass_ahead_pct", "Pass Ahead %", "Distributions going forward.", "percentage"),
    MetricInfo("avg_targeted_xthreat", "Targeted xThreat", "Expected threat created by passes.", "number"),
    MetricInfo("avg_safe_dangerous_options", "Safe Dangerous Options", "Available safe but dangerous options.", "number"),
    MetricInfo("forward_momentum_pct", "Forward Momentum %", "Distributions with forward momentum.", "percentage"),
    # Distribution context metrics
    MetricInfo("avg_distribution_speed", "Distribution Speed", "Seconds from possession start to distribution. Lower = faster.", "number"),
    MetricInfo("quick_counter_launch_pct", "Counter Launch %", "Distributions that launch quick breaks.", "percentage"),
    MetricInfo("distribution_attack_rate", "Attack Rate %", "Distributions leading to shots or goals.", "percentage"),
)

MIDFIELDER_METRICS = (
    # Ball Progression
    MetricInfo("progressive_pass_pct", "Progressive Pass %", "Percentage of passes that progress play forward.", "percentage"),
    MetricInfo("progressive_carry_pct", "Progressive Carry %", "Percentage of carries moving ball forward.", "percentage"),
    MetricInfo("final_third_pass_pct", "Final Third Pass %", "Percentage of passes into the attacking third.", "percentage"),
    MetricInfo("avg_pass_distance", "Pass Distance", "Average pass distance in meters.", "distance"),
    # Defensive Contribution
    MetricInfo("pressing_rate", "Pressing %", "How often midfielder engages proactively.", "percentage"),
    MetricInfo("tackle_success_rate", "Tackle Success %", "Percentage of successful defensive engagements.", "percentage"),
    MetricInfo("interception_rate", "Interception Rate", "Interceptions per possession event.", "number"),
    MetricInfo("ball_recovery_rate", "Ball Recovery Rate", "Ball recoveries per possession event.", "number"),
    # Creativity
    MetricInfo("key_pass_rate", "Key Pass %", "Percentage of passes leading to shots.", "percentage"),
    MetricInfo("through_ball_pct", "Through Ball %", "Percentage of through ball passes.", "percentage"),
    MetricInfo("danger_creation_rate", "Danger Creation %", "Possessions creating danger.", "percentage"),
    # Work Rate & Positioning
    MetricInfo("central_presence_pct", "Central %", "Percentage of events in central zones.", "percentage"),
    MetricInfo("attacking_third_pct", "Attacking Third %", "Time spent in attacking third.", "percentage"),
    MetricInfo("avg_speed", "Avg Speed", "Average speed during actions (m/s).", "speed"),
    # Extra
    MetricInfo("pass_accuracy", "Pass Accuracy %", "Percentage of completed passes.", "percentage"),
)

POSITION_CONFIGS: dict[str, PositionConfig] = {
    "forward": PositionConfig(
        name="Forward",
        metrics=FORWARD_METRICS,
        count_field="total_entries",
        criteria="Forwards create danger through intelligent movement, finding space, and clinical finishing. Key: high danger rate, good separation, effective central positioning.",
    ),
    "defender": PositionConfig(
        name="Defender",
        metrics=DEFENDER_METRICS,
        count_field="total_engagements",
        criteria="Defenders stop attacks, maintain goal-side positioning, and press intelligently. Key: high stop/reduce rates, consistent positioning, appropriate engagement distance.",
    ),
    "goalkeeper": PositionConfig(
        name="Goalkeeper",
        metrics=GOALKEEPER_METRICS,
        count_field="total_distributions",
        criteria="Goalkeepers distribute effectively and make good decisions. Key: high pass success, appropriate distance for style, quick distribution.",
    ),
    "midfielder": PositionConfig(
        name="Midfielder",
        metrics=MIDFIELDER_METRICS,
        count_field="total_events",
        criteria="Midfielders control tempo, progress play, contribute defensively, and create chances. Key: high progressive pass %, good pressing rate, creativity in key passes.",
    ),
}


# =============================================================================
# Data Classes for Insights
# =============================================================================

@dataclass(frozen=True, slots=True)
class DevelopmentGap:
    """A gap between player metric and archetype target."""

    metric_key: str
    metric_label: str
    player_value: float
    target_value: float
    gap: float  # Absolute difference (normalised 0-100 scale)
    direction: str  # "increase" or "decrease"
    weight: float  # How important this metric is

    @property
    def priority_score(self) -> float:
        """Higher score = more important to develop (gap * weight)."""
        return self.gap * self.weight

    def format(self) -> str:
        """Format as readable string."""
        arrow = "↑" if self.direction == "increase" else "↓"
        return f"{self.metric_label}: {self.player_value:.0f} → {self.target_value:.0f} ({arrow} {self.gap:.0f} gap, {self.weight*100:.0f}% weight)"


@dataclass(frozen=True, slots=True)
class ArchetypeFit:
    """Similarity to a specific archetype."""

    archetype_name: str
    similarity: float
    position: str

    def format(self) -> str:
        """Format as readable string."""
        return f"{self.archetype_name}: {self.similarity:.1f}%"


@dataclass(frozen=True, slots=True)
class ConfidenceLevel:
    """Confidence indicator based on sample size."""

    level: str  # "High", "Medium", "Low"
    sample_size: int
    description: str
    emoji: str

    @classmethod
    def from_sample_size(cls, sample_size: int) -> "ConfidenceLevel":
        """Create confidence level from sample size."""
        if sample_size >= 10:
            return cls("High", sample_size, "Reliable profile based on sufficient data", "🟢")
        elif sample_size >= 5:
            return cls("Medium", sample_size, "Reasonable estimate, more data would help", "🟡")
        else:
            return cls("Low", sample_size, "Preliminary profile, needs more observations", "🔴")

    def format(self) -> str:
        """Format as readable string."""
        return f"{self.emoji} {self.level} confidence ({self.sample_size} samples): {self.description}"


@dataclass
class PlayerInsightData:
    """Enriched player data for AI insights."""

    name: str
    team: str
    age: int | None
    similarity: float
    sample_size: int
    metrics: dict[str, float]
    percentiles: dict[str, float]
    confidence: ConfidenceLevel | None = None
    development_gaps: list[DevelopmentGap] = field(default_factory=list)
    archetype_fits: list[ArchetypeFit] = field(default_factory=list)
    similar_players: list[str] = field(default_factory=list)
    age_percentiles: dict[str, float] = field(default_factory=dict)

    def format_metric(self, key: str, metric_info: MetricInfo) -> str:
        """Format a single metric with percentile context."""
        value = self.metrics.get(key)
        percentile = self.percentiles.get(key)

        if value is None:
            return "N/A"

        # Format value based on type
        if metric_info.format_type == "percentage":
            formatted = f"{value:.1f}%"
        elif metric_info.format_type == "distance":
            formatted = f"{value:.1f}m"
        elif metric_info.format_type == "speed":
            formatted = f"{value:.1f}m/s"
        else:
            formatted = f"{value:.1f}"

        # Add percentile if available
        if percentile is not None:
            return f"{formatted} (P{percentile:.0f})"
        return formatted

    def format_top_gaps(self, n: int = 3) -> str:
        """Format top N development gaps by priority."""
        if not self.development_gaps:
            return "No gaps identified"
        sorted_gaps = sorted(self.development_gaps, key=lambda g: g.priority_score, reverse=True)
        return "; ".join(g.format() for g in sorted_gaps[:n])

    def format_archetype_fits(self) -> str:
        """Format archetype fits as string."""
        if not self.archetype_fits:
            return "N/A"
        return ", ".join(f.format() for f in self.archetype_fits)

    def format_similar_players(self) -> str:
        """Format similar players as string."""
        if not self.similar_players:
            return "None identified"
        return ", ".join(self.similar_players)


@dataclass
class ArchetypeInsightData:
    """Archetype data for AI insights."""

    name: str
    description: str
    targets: dict[str, float]
    weights: dict[str, float]

    def format_weights(self, metrics: list[str]) -> str:
        """Format weights as a readable string."""
        parts = []
        for key in metrics:
            weight = self.weights.get(key, 0)
            if weight > 0:
                parts.append(f"{key}: {weight*100:.0f}%")
        return ", ".join(parts) if parts else "Equal weights"

    def format_targets(self, metrics: list[str]) -> str:
        """Format target values as a readable string."""
        parts = []
        for key in metrics:
            target = self.targets.get(key)
            if target is not None:
                parts.append(f"{key}: {target:.0f}")
        return ", ".join(parts) if parts else "N/A"


# =============================================================================
# Player Analyzer Class
# =============================================================================

class PlayerAnalyzer:
    """Analyze players for enhanced scouting insights.

    Provides:
    - Development gap analysis (metrics furthest from target)
    - Multi-archetype fit comparison
    - Age-based percentile rankings
    - Similar player identification
    - Confidence level assessment
    """

    # Archetypes by position for multi-fit comparison
    POSITION_ARCHETYPES: dict[str, list[str]] = {
        "forward": ["alvarez", "giroud", "kane", "lewandowski", "rashford", "en_nesyri"],
        "defender": ["gvardiol", "vandijk", "hakimi"],
        "goalkeeper": ["neuer", "lloris", "bounou"],
        "midfielder": ["enzo", "tchouameni", "depaul", "griezmann", "pedri", "bellingham"],
    }

    def __init__(self, position_type: str = "forward"):
        """Initialize analyzer for a position type."""
        self.position_type = position_type
        self.config = POSITION_CONFIGS.get(position_type, POSITION_CONFIGS["forward"])

    def compute_development_gaps(
        self,
        player_metrics: dict[str, float],
        archetype: ArchetypeInsightData,
    ) -> list[DevelopmentGap]:
        """Identify gaps between player metrics and archetype targets.

        Returns gaps sorted by priority (gap size * weight).
        """
        gaps = []
        for metric_info in self.config.metrics:
            key = metric_info.key
            player_val = player_metrics.get(key)
            target_val = archetype.targets.get(key)
            weight = archetype.weights.get(key, 0.1)

            if player_val is None or target_val is None:
                continue

            gap = abs(target_val - player_val)
            direction = "increase" if target_val > player_val else "decrease"

            gaps.append(DevelopmentGap(
                metric_key=key,
                metric_label=metric_info.label,
                player_value=player_val,
                target_value=target_val,
                gap=gap,
                direction=direction,
                weight=weight,
            ))

        # Sort by priority score (gap * weight)
        return sorted(gaps, key=lambda g: g.priority_score, reverse=True)

    def compute_multi_archetype_fit(
        self,
        player_profile: dict[str, float],
        all_profiles: pl.DataFrame,
    ) -> list[ArchetypeFit]:
        """Compare player against all archetypes for their position.

        Uses lazy imports to avoid circular dependencies.
        """
        from src.core.archetype import Archetype
        from src.core.similarity import SimilarityEngine

        fits = []
        archetype_keys = self.POSITION_ARCHETYPES.get(self.position_type, [])

        for arch_key in archetype_keys:
            try:
                archetype = Archetype.from_statsbomb(arch_key)
                engine = SimilarityEngine(archetype)
                engine.fit(all_profiles)

                # Get similarity for this specific player
                player_name = player_profile.get("player_name", "")
                if player_name:
                    # Find player in rankings
                    rankings = engine.rank(top_n=len(all_profiles))
                    player_row = rankings.filter(pl.col("player_name") == player_name)
                    if len(player_row) > 0:
                        similarity = player_row["similarity_score"][0]
                        fits.append(ArchetypeFit(
                            archetype_name=archetype.name.replace("_", " ").title(),
                            similarity=similarity,
                            position=self.position_type,
                        ))
            except Exception:
                continue  # Skip if archetype fails to load

        # Sort by similarity descending
        return sorted(fits, key=lambda f: f.similarity, reverse=True)

    def compute_age_percentiles(
        self,
        player_age: int | None,
        player_metrics: dict[str, float],
        all_profiles: pl.DataFrame,
    ) -> dict[str, float]:
        """Compute percentile ranks among similar age group.

        Age groups: U21, U23, U25, 25+
        """
        if player_age is None or "age" not in all_profiles.columns:
            return {}

        # Define age group
        if player_age < 21:
            age_filter = pl.col("age") < 21
            group_name = "U21"
        elif player_age < 23:
            age_filter = (pl.col("age") >= 21) & (pl.col("age") < 23)
            group_name = "U23"
        elif player_age < 25:
            age_filter = (pl.col("age") >= 23) & (pl.col("age") < 25)
            group_name = "U25"
        else:
            age_filter = pl.col("age") >= 25
            group_name = "25+"

        # Filter to age group
        age_group = all_profiles.filter(age_filter)
        if len(age_group) < 3:  # Need at least 3 players for meaningful percentile
            return {}

        percentiles = {}
        for key, value in player_metrics.items():
            if key not in age_group.columns or value is None:
                continue
            all_values = age_group[key].drop_nulls().to_numpy()
            if len(all_values) == 0:
                continue
            pct = (np.sum(all_values <= value) / len(all_values)) * 100
            percentiles[f"{key}_age_group"] = pct

        # Add group name for context
        percentiles["_age_group"] = group_name
        percentiles["_age_group_size"] = len(age_group)

        return percentiles

    def find_similar_players(
        self,
        player_name: str,
        all_profiles: pl.DataFrame,
        metric_keys: list[str],
        top_n: int = 3,
    ) -> list[str]:
        """Find players with similar profiles using cosine similarity.

        Returns player names (excluding the input player).
        """
        if player_name not in all_profiles["player_name"].to_list():
            return []

        # Get player's metrics
        player_row = all_profiles.filter(pl.col("player_name") == player_name)
        if len(player_row) == 0:
            return []

        # Extract metric columns that exist
        available_metrics = [k for k in metric_keys if k in all_profiles.columns]
        if not available_metrics:
            return []

        # Compute z-scores for all players
        normalized = all_profiles.select(
            ["player_name"] + [
                ((pl.col(m) - pl.col(m).mean()) / pl.col(m).std()).alias(f"{m}_z")
                for m in available_metrics
            ]
        )

        # Get target player's z-scores
        target_row = normalized.filter(pl.col("player_name") == player_name)
        if len(target_row) == 0:
            return []

        # Extract z-scores, handling None values
        target_z_list = []
        for m in available_metrics:
            val = target_row[f"{m}_z"][0]
            if val is None:
                target_z_list.append(np.nan)
            else:
                target_z_list.append(float(val))
        target_z = np.array(target_z_list, dtype=np.float64)

        # Handle NaN in target
        if np.any(np.isnan(target_z)):
            return []

        # Compute cosine similarity with all other players
        similarities = []
        for row in normalized.filter(pl.col("player_name") != player_name).to_dicts():
            other_z = np.array([float(row.get(f"{m}_z", np.nan) or np.nan) for m in available_metrics], dtype=np.float64)
            if np.any(np.isnan(other_z)):
                continue

            # Cosine similarity
            dot = np.dot(target_z, other_z)
            norm = np.linalg.norm(target_z) * np.linalg.norm(other_z)
            if norm > 0:
                sim = dot / norm
                similarities.append((row["player_name"], sim))

        # Sort by similarity and return top N names
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:top_n]]

    def get_confidence_level(self, sample_size: int) -> ConfidenceLevel:
        """Get confidence level based on sample size."""
        return ConfidenceLevel.from_sample_size(sample_size)


# =============================================================================
# Insight Generator Class
# =============================================================================

class InsightGenerator:
    """Generate AI-powered scouting insights.

    Uses position-specific prompts enriched with:
    - Feature weights (what matters most)
    - Archetype targets (ideal values)
    - Percentile ranks (context vs dataset)
    - Sample size warnings (reliability)
    - Domain knowledge (player/team/league context from model's training)
    - Development gaps (areas to improve)
    - Multi-archetype fit (similarity to other archetypes)
    - Age-based percentiles (comparison within age group)
    - Similar players (alternatives in the dataset)
    """

    SYSTEM_PROMPT = """You are a football scout and analyst writing player similarity reports.
Use Australian English (defence, centre, metre). Be practical and specific.
Avoid vague generalities and overstatements. Focus on actionable insights.

You have knowledge about football players, leagues, and playing styles. When you recognise
a player or team, incorporate relevant context: their background, career trajectory,
known strengths, or how the A-League compares to other leagues they may have played in.
The A-League is Australia's top professional league, competitive but below Europe's top 5."""

    MIN_RELIABLE_SAMPLES = 5  # Below this, add reliability warning

    def __init__(self, position_type: str = "forward", enable_enhanced_analysis: bool = True):
        """Initialize generator for a position type.

        Args:
            position_type: One of "forward", "defender", "goalkeeper"
            enable_enhanced_analysis: If True, compute development gaps, multi-archetype fit, etc.
                                     Set to False for faster generation with basic analysis.
        """
        self.position_type = position_type
        self.config = POSITION_CONFIGS.get(position_type, POSITION_CONFIGS["forward"])
        self.enable_enhanced_analysis = enable_enhanced_analysis
        self.analyzer = PlayerAnalyzer(position_type) if enable_enhanced_analysis else None

    def compute_percentiles(
        self,
        player_values: dict[str, float],
        all_profiles: pl.DataFrame
    ) -> dict[str, float]:
        """Compute percentile ranks for player metrics against full dataset."""
        percentiles = {}
        for key, value in player_values.items():
            if key not in all_profiles.columns or value is None:
                continue
            all_values = all_profiles[key].drop_nulls().to_numpy()
            if len(all_values) == 0:
                continue
            # Compute percentile rank
            percentile = (np.sum(all_values <= value) / len(all_values)) * 100
            percentiles[key] = percentile
        return percentiles

    def extract_player_data(
        self,
        row: dict,
        all_profiles: pl.DataFrame,
        archetype_data: ArchetypeInsightData | None = None,
    ) -> PlayerInsightData:
        """Extract enriched player data from a row.

        Args:
            row: Dictionary with player data
            all_profiles: Full dataset for percentile computation
            archetype_data: Archetype for development gap analysis (optional)
        """
        from src.data.player_ages import get_player_age

        name = row.get("player_name", "Unknown")
        metrics = {m.key: row.get(m.key) for m in self.config.metrics}
        sample_size = row.get(self.config.count_field, 0)
        age = row.get("age") or get_player_age(name)

        # Basic data
        player_data = PlayerInsightData(
            name=name,
            team=row.get("team_name", ""),
            age=age,
            similarity=row.get("similarity_score", 0),
            sample_size=sample_size,
            metrics=metrics,
            percentiles=self.compute_percentiles(metrics, all_profiles),
        )

        # Enhanced analysis (if enabled and analyzer available)
        if self.analyzer and self.enable_enhanced_analysis:
            # Confidence level
            player_data.confidence = self.analyzer.get_confidence_level(sample_size)

            # Development gaps (requires archetype)
            if archetype_data:
                player_data.development_gaps = self.analyzer.compute_development_gaps(
                    metrics, archetype_data
                )

            # Similar players in dataset
            metric_keys = self.config.get_metric_keys()
            player_data.similar_players = self.analyzer.find_similar_players(
                name, all_profiles, metric_keys, top_n=3
            )

            # Age-based percentiles
            if age:
                player_data.age_percentiles = self.analyzer.compute_age_percentiles(
                    age, metrics, all_profiles
                )

        return player_data

    def extract_archetype_data(self, archetype: Any) -> ArchetypeInsightData:
        """Extract archetype data from an Archetype object."""
        if hasattr(archetype, "name") and hasattr(archetype, "description"):
            name = archetype.name.replace("_", " ").title()
            description = archetype.description
            targets = getattr(archetype, "target_profile", {})
            weights = getattr(archetype, "weights", {})
        else:
            # Backwards compatibility
            name = "Julian Alvarez"
            description = str(archetype)
            targets = {}
            weights = {}

        return ArchetypeInsightData(
            name=name,
            description=description,
            targets=targets,
            weights=weights,
        )

    def format_player_summary(self, player: PlayerInsightData) -> str:
        """Format a single player for the prompt with enhanced analysis."""
        age_str = f"Age {player.age}" if player.age else "Age unknown"

        # Format metrics with percentiles
        metric_parts = []
        for metric_info in self.config.metrics:
            formatted = player.format_metric(metric_info.key, metric_info)
            metric_parts.append(f"{metric_info.label}: {formatted}")

        # Confidence indicator
        confidence_str = ""
        if player.confidence:
            confidence_str = f" [{player.confidence.level} confidence]"
        elif player.sample_size < self.MIN_RELIABLE_SAMPLES:
            confidence_str = f" [LOW SAMPLE: {player.sample_size}]"

        # Build base summary
        lines = [
            f"- **{player.name}** ({player.team}): {age_str}, "
            f"Similarity {player.similarity:.1f}%, {player.sample_size} samples{confidence_str}",
            f"  Metrics: {', '.join(metric_parts)}",
        ]

        # Add development gaps (top 2 priority areas)
        if player.development_gaps:
            top_gaps = sorted(player.development_gaps, key=lambda g: g.priority_score, reverse=True)[:2]
            gap_strs = [f"{g.metric_label} ({g.direction} by {g.gap:.0f})" for g in top_gaps]
            lines.append(f"  Development areas: {', '.join(gap_strs)}")

        # Add similar players
        if player.similar_players:
            lines.append(f"  Similar profiles: {', '.join(player.similar_players)}")

        # Add age group context
        if player.age_percentiles and "_age_group" in player.age_percentiles:
            group = player.age_percentiles["_age_group"]
            size = int(player.age_percentiles.get("_age_group_size", 0))
            lines.append(f"  Age group: {group} ({size} players in group)")

        return "\n".join(lines)

    def compute_dataset_stats(self, profiles: pl.DataFrame) -> str:
        """Compute dataset statistics for context."""
        parts = []
        for metric_info in self.config.metrics:
            if metric_info.key in profiles.columns:
                col = profiles[metric_info.key].drop_nulls()
                if len(col) > 0:
                    avg = col.mean()
                    std = col.std()
                    if avg is not None and std is not None:
                        parts.append(f"{metric_info.label}: {avg:.1f} (±{std:.1f})")
                    elif avg is not None:
                        parts.append(f"{metric_info.label}: {avg:.1f}")
        return ", ".join(parts) if parts else "N/A"

    def format_metrics_glossary(self) -> str:
        """Format all metrics with their definitions as a glossary."""
        lines = []
        for metric_info in self.config.metrics:
            lines.append(f"- **{metric_info.label}** ({metric_info.key}): {metric_info.definition}")
        return "\n".join(lines)

    def build_prompt(
        self,
        players: list[PlayerInsightData],
        archetype: ArchetypeInsightData,
        total_players: int,
        total_events: int,
        dataset_stats: str,
    ) -> str:
        """Build the enriched prompt for AI generation."""
        metric_keys = self.config.get_metric_keys()

        player_summaries = "\n".join(
            self.format_player_summary(p) for p in players
        )

        # Build confidence summary
        confidence_summary = self._build_confidence_summary(players)

        return f"""Analyse these A-League {self.config.name.lower()}s compared to the {archetype.name} archetype.

ANALYSIS CONTEXT:
- Position: {self.config.name}
- Dataset: {total_events:,} {self.config.count_field.replace('total_', '')} from {total_players} players (10 A-League matches)
- Method: Weighted cosine similarity on z-score normalised features
- Percentiles shown as (Pxx) indicate rank vs all {total_players} players
{confidence_summary}

FEATURE WEIGHTS (what matters most in similarity):
{archetype.format_weights(metric_keys)}

{archetype.name.upper()} ARCHETYPE:
{archetype.description}

TARGET PROFILE (ideal values):
{archetype.format_targets(metric_keys)}

WHAT MAKES A GREAT {self.config.name.upper()}:
{self.config.criteria}

METRICS GLOSSARY (definitions for all metrics):
{self.format_metrics_glossary()}

TOP CANDIDATES (ranked by similarity):
{player_summaries}

DATASET AVERAGES (mean ± std):
{dataset_stats}

ANALYSIS INSTRUCTIONS:
Write 4-5 concise paragraphs covering:

1. **Best Match**: Which candidate best matches {archetype.name}'s style? Reference specific metrics, percentiles, and confidence levels.

2. **Development Areas**: For the top candidate(s), what are the priority development areas? Use the gap analysis provided (metric name, direction, gap size, weight).

3. **Alternative Profiles**: Note the "Similar profiles" suggestions - are there backup candidates with comparable styles? Comment on any patterns.

4. **Scouting Recommendation**: Considering age, development potential, confidence level, and age group rankings, who deserves priority observation? Be specific about why.

5. **Domain Context**: If you recognise any players, teams, or can add context about {archetype.name}'s actual playing style, include it. Mention international experience, career trajectory, or A-League context.

CONFIDENCE LEVELS:
- High: 10+ samples, reliable profile
- Medium: 5-9 samples, reasonable estimate
- Low: <5 samples, preliminary data only

DOMAIN KNOWLEDGE: Use your knowledge of {archetype.name}, the A-League, and any players/teams you recognise. The A-League is Australia's top league, competitive but below Europe's top 5.

FORMATTING: Player names are already bold. Be specific and practical. Australian English."""

    def _build_confidence_summary(self, players: list[PlayerInsightData]) -> str:
        """Build summary of confidence levels across candidates."""
        if not players or not players[0].confidence:
            return ""

        counts = {"High": 0, "Medium": 0, "Low": 0}
        for p in players:
            if p.confidence:
                counts[p.confidence.level] += 1

        parts = []
        for level, count in counts.items():
            if count > 0:
                parts.append(f"{count} {level}")

        if parts:
            return f"- Confidence breakdown: {', '.join(parts)}"
        return ""

    def generate(
        self,
        ranked_players: pl.DataFrame,
        archetype: Any,
        top_n: int = 5,
        model: str | None = None,
    ) -> str:
        """Generate AI insight for top candidates.

        Args:
            ranked_players: DataFrame with ranked player profiles
            archetype: Archetype object with name, description, targets, weights
            top_n: Number of candidates to analyse
            model: Model key to use (default: auto-select based on available tokens)

        Returns:
            AI-generated scouting insight as markdown
        """
        # Extract archetype data first (needed for development gaps)
        archetype_data = self.extract_archetype_data(archetype)

        # Extract player data with enhanced analysis
        players = [
            self.extract_player_data(row, ranked_players, archetype_data)
            for row in ranked_players.head(top_n).to_dicts()
        ]

        # Compute stats
        total_players = len(ranked_players)
        count_field = self.config.count_field
        total_events = int(ranked_players[count_field].sum()) if count_field in ranked_players.columns else 0
        dataset_stats = self.compute_dataset_stats(ranked_players)

        # Build prompt
        prompt = self.build_prompt(
            players=players,
            archetype=archetype_data,
            total_players=total_players,
            total_events=total_events,
            dataset_stats=dataset_stats,
        )

        
        return _call_model(prompt, model, max_tokens=2000)


# =============================================================================
# Token Management
# =============================================================================

def get_github_token() -> str | None:
    """Load GitHub token from file or environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    token_path = Path(__file__).parent.parent.parent / "github_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def get_hf_token() -> str | None:
    """Load HF token from environment or file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path(__file__).parent.parent.parent / "hf_token.txt"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def set_github_token(token: str) -> None:
    """Set GitHub token in environment."""
    os.environ["GITHUB_TOKEN"] = token


def set_hf_token(token: str) -> None:
    """Set HF token in environment."""
    os.environ["HF_TOKEN"] = token


def get_default_model() -> str:
    """Get default model based on available tokens."""
    if get_github_token():
        return DEFAULT_GITHUB_MODEL
    if get_hf_token():
        return DEFAULT_HF_MODEL
    return DEFAULT_GITHUB_MODEL


def get_available_backend() -> str | None:
    """Return which backend is available."""
    if get_github_token():
        return "github"
    if get_hf_token():
        return "huggingface"
    return None


def get_model_backend(model_key: str) -> str:
    """Return which backend a model uses."""
    config = MODELS.get(model_key)
    return config.backend if config else "github"


def has_valid_token() -> bool:
    """Check if any valid token exists."""
    gh = get_github_token()
    hf = get_hf_token()
    return (gh and len(gh) > 10) or (hf and len(hf) > 10)


def get_available_models() -> list[tuple[str, str]]:
    """Get available models based on tokens."""
    models = []
    if get_github_token() and len(get_github_token()) > 10:
        for key in GITHUB_MODELS:
            models.append((key, MODEL_DISPLAY_NAMES[key]))
    if get_hf_token() and len(get_hf_token()) > 10:
        for key in HF_MODELS:
            models.append((key, MODEL_DISPLAY_NAMES[key]))
    return models


# =============================================================================
# Model Calling Functions
# =============================================================================

def _call_github_model(prompt: str, model_key: str, max_tokens: int = 500) -> str:
    """Call GitHub Models API."""
    token = get_github_token()
    if not token:
        return "AI insights unavailable (no GitHub token)"

    config = MODELS.get(model_key, MODELS[DEFAULT_GITHUB_MODEL])
    effective_max_tokens = max(max_tokens, config.max_tokens)

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(token),
        )

        response = client.complete(
            messages=[
                SystemMessage(InsightGenerator.SYSTEM_PROMPT),
                UserMessage(prompt),
            ],
            temperature=config.temperature,
            top_p=1.0,
            max_tokens=effective_max_tokens,
            model=config.model_id,
        )

        return response.choices[0].message.content or f"Empty response from {config.model_id}"

    except ImportError:
        import requests

        try:
            response = requests.post(
                "https://models.github.ai/inference/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.model_id,
                    "messages": [
                        {"role": "system", "content": InsightGenerator.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": effective_max_tokens,
                    "temperature": config.temperature,
                    "top_p": 1.0,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or "Empty response"
        except Exception as e:
            return f"AI unavailable: {e}"

    except Exception as e:
        return f"AI unavailable: {e}"


def _call_hf_model(prompt: str, model_id: str, max_tokens: int = 500) -> str:
    """Call HuggingFace Inference API."""
    token = get_hf_token()
    if not token:
        return "AI insights unavailable (no HF token)"

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token)
        response = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": InsightGenerator.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or "Empty response"

    except ImportError:
        import requests

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": InsightGenerator.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"AI unavailable: {e}"

    except Exception as e:
        return f"AI unavailable: {e}"


def _call_model(prompt: str, model_key: str | None = None, max_tokens: int = 500) -> str:
    """Route to appropriate backend."""
    if model_key is None:
        model_key = get_default_model()

    config = MODELS.get(model_key)
    if config and config.backend == "huggingface":
        return _call_hf_model(prompt, config.model_id, max_tokens)
    return _call_github_model(prompt, model_key, max_tokens)


# =============================================================================
# Public API (backwards compatible)
# =============================================================================

def generate_similarity_insight(
    ranked_players: pl.DataFrame,
    archetype: Any,
    top_n: int = 5,
    model: str | None = None,
    position_type: str = "forward",
) -> str:
    """Generate AI insight comparing top candidates to an archetype.

    This is the main public function for generating insights.

    Args:
        ranked_players: DataFrame with ranked player profiles
        archetype: Archetype object with name, description, targets, weights
        top_n: Number of top candidates to include
        model: Model to use for generation
        position_type: One of "forward", "defender", "goalkeeper"

    Returns:
        AI-generated scouting insight as markdown string
    """
    generator = InsightGenerator(position_type)
    return generator.generate(ranked_players, archetype, top_n, model)


def generate_player_report(
    player_name: str,
    player_profile: dict,
    similarity_score: float,
    archetype: Any,
    model: str | None = None,
    position_type: str = "forward",
) -> str:
    """Generate detailed scouting report for a single player."""
    generator = InsightGenerator(position_type)
    archetype_data = generator.extract_archetype_data(archetype)
    config = generator.config

    # Format metrics
    metrics_lines = []
    for metric_info in config.metrics:
        value = player_profile.get(metric_info.key)
        if value is not None:
            if metric_info.format_type == "percentage":
                formatted = f"{value:.1f}%"
            elif metric_info.format_type == "distance":
                formatted = f"{value:.1f}m"
            elif metric_info.format_type == "speed":
                formatted = f"{value:.1f}m/s"
            else:
                formatted = f"{value:.1f}"
            metrics_lines.append(f"- {metric_info.label}: {formatted} ({metric_info.definition})")

    count = player_profile.get(config.count_field, 0)
    metrics_lines.append(f"- Sample size: {count} {config.count_field.replace('total_', '')}")

    team_name = player_profile.get('team_name', 'Unknown')
    prompt = f"""Write a brief scouting report comparing {player_name} to {archetype_data.name}.

{archetype_data.name.upper()} ARCHETYPE:
{archetype_data.description}

PLAYER PROFILE:
- Name: {player_name}
- Team: {team_name}
- Similarity Score: {similarity_score:.1f}%
{chr(10).join(metrics_lines)}

Write 2-3 paragraphs:
1. Strengths matching {archetype_data.name}'s style
2. Areas of difference from the archetype
3. If you recognise {player_name} or {team_name}, add context about their background, career trajectory, or known attributes beyond the data

Use your knowledge of the A-League, {archetype_data.name}, and {player_name} if available.
Be specific and practical. Australian English."""

    return _call_model(prompt, model, max_tokens=2000)
