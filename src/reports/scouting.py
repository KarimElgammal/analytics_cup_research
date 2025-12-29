"""Scouting Report PDF Generation."""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

from src.reports.components import (
    Colors,
    FooterComponent,
    HeaderComponent,
    InsightComponent,
    MetricsTableComponent,
    RadarChartComponent,
)


@dataclass
class PlayerData:
    """Player data for report generation."""

    name: str
    team: str
    age: int | None
    position: str
    similarity: float
    metrics: dict[str, float]
    percentiles: dict[str, float]
    sample_size: int = 0


@dataclass
class ArchetypeData:
    """Archetype data for report generation."""

    name: str
    description: str
    target_profile: dict[str, float] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    include_radar: bool = True
    include_metrics: bool = True
    include_insight: bool = True
    radar_features: list[str] = field(default_factory=list)
    metric_labels: dict[str, str] = field(default_factory=dict)
    model_name: str | None = None


class ScoutingReport:
    """Generate PDF scouting reports for players.

    Usage:
        report = ScoutingReport(player_data, archetype_data)
        pdf_bytes = report.generate()

        # Or save to file
        report.save("report.pdf")
    """

    def __init__(
        self,
        player: PlayerData,
        archetype: ArchetypeData,
        config: ReportConfig | None = None,
        insight: str | None = None,
        all_profiles: Any = None,  # polars DataFrame for radar normalization
    ):
        self.player = player
        self.archetype = archetype
        self.config = config or ReportConfig()
        self.insight = insight
        self.all_profiles = all_profiles
        self._pdf = None
        self._temp_files: list[Path] = []

    def generate(self) -> bytes:
        """Generate the PDF report and return as bytes."""
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError(
                "fpdf2 is required for PDF generation. "
                "Install with: pip install fpdf2"
            )

        self._pdf = FPDF()
        self._pdf.set_auto_page_break(auto=True, margin=20)
        self._pdf.add_page()

        # Render components
        self._render_header()

        if self.config.include_radar and self.config.radar_features:
            self._render_radar()

        if self.config.include_metrics:
            self._render_metrics()

        if self.config.include_insight and self.insight:
            self._pdf.add_page()  # Insight on separate page
            self._render_insight()

        # Disable auto page break before footer to prevent extra empty page
        self._pdf.set_auto_page_break(auto=False)
        self._render_footer()

        # Get PDF bytes
        pdf_bytes = self._pdf.output()

        # Cleanup temp files
        self._cleanup()

        return bytes(pdf_bytes)

    def save(self, path: str | Path) -> None:
        """Generate and save PDF to file."""
        pdf_bytes = self.generate()
        Path(path).write_bytes(pdf_bytes)

    def _render_header(self) -> None:
        """Render the header section."""
        header = HeaderComponent(self._pdf)
        header.render(
            player_name=self.player.name,
            team=self.player.team,
            age=self.player.age,
            position=self.player.position,
            archetype=self.archetype.name,
            similarity=self.player.similarity,
        )

    def _render_radar(self) -> None:
        """Render the radar chart section."""
        radar_path = self._generate_radar_image()
        if radar_path:
            radar = RadarChartComponent(self._pdf)
            radar.render(str(radar_path), width=120)

    def _render_metrics(self) -> None:
        """Render the metrics table section."""
        metrics = MetricsTableComponent(self._pdf)
        metrics.render(
            metrics=self.player.metrics,
            percentiles=self.player.percentiles,
            metric_labels=self.config.metric_labels,
        )

    def _render_insight(self) -> None:
        """Render the AI insight section."""
        insight = InsightComponent(self._pdf)
        insight.render(
            insight=self.insight,
            model_name=self.config.model_name,
        )

    def _render_footer(self) -> None:
        """Render the footer on all pages."""
        total_pages = self._pdf.page_no()
        for page_num in range(1, total_pages + 1):
            self._pdf.page = page_num
            footer = FooterComponent(self._pdf)
            footer.render(page_num=page_num, total_pages=total_pages)

    def _generate_radar_image(self) -> Path | None:
        """Generate radar chart and save to temp file."""
        if not self.config.radar_features:
            return None

        try:
            from src.visualization.radar import plot_radar_comparison, normalize_for_radar_percentile
        except ImportError:
            return None

        features = self.config.radar_features

        # Prepare player data for radar
        player_dict = {"name": f"{self.player.name} ({self.player.team})"}

        # Normalize if we have all_profiles
        if self.all_profiles is not None:
            import polars as pl

            # Create a single-row DataFrame for the player (filter out None values)
            player_row = {
                "player_name": self.player.name,
                "team_name": self.player.team,
            }
            for k, v in self.player.metrics.items():
                if v is not None:
                    player_row[k] = v

            player_df = pl.DataFrame([player_row])
            norm_df = normalize_for_radar_percentile(player_df, self.all_profiles, features)

            for f in features:
                norm_key = f"{f}_norm"
                if norm_key in norm_df.columns:
                    val = norm_df[norm_key][0]
                    player_dict[f] = val if val is not None else 50
                elif f in self.player.metrics and self.player.metrics[f] is not None:
                    pctl = self.player.percentiles.get(f)
                    player_dict[f] = pctl if pctl is not None else 50
                else:
                    player_dict[f] = 50  # Default for missing values
        else:
            # Use percentiles as fallback
            for f in features:
                pctl = self.player.percentiles.get(f)
                player_dict[f] = pctl if pctl is not None else 50

        # Target profile (ensure no None values)
        target_profile = {}
        for f in features:
            val = self.archetype.target_profile.get(f)
            target_profile[f] = val if val is not None else 50

        # Generate radar chart
        fig = plot_radar_comparison(
            players=[player_dict],
            features=features,
            title="",
            include_alvarez=True,
            target_profile=target_profile,
            target_name=self.archetype.name,
        )

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        self._temp_files.append(temp_path)

        fig.savefig(temp_path, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

        return temp_path

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        for path in self._temp_files:
            try:
                path.unlink()
            except OSError:
                pass
        self._temp_files.clear()


def generate_scouting_report(
    player_row: dict,
    archetype: Any,
    all_profiles: Any,
    position_type: str,
    insight: str | None = None,
    model_name: str | None = None,
) -> bytes:
    """Convenience function to generate a scouting report.

    Args:
        player_row: Dictionary with player data (from polars row)
        archetype: Archetype object with name, description, target_profile
        all_profiles: Full profiles DataFrame for percentile computation
        position_type: "forward", "defender", or "goalkeeper"
        insight: Optional AI-generated insight text
        model_name: Optional model name for attribution

    Returns:
        PDF as bytes
    """
    from src.utils.ai_insights import POSITION_CONFIGS

    config = POSITION_CONFIGS.get(position_type)
    if not config:
        raise ValueError(f"Unknown position type: {position_type}")

    # Extract player data
    player = PlayerData(
        name=player_row.get("player_name", "Unknown"),
        team=player_row.get("team_name", ""),
        age=player_row.get("age"),
        position=position_type.title(),
        similarity=player_row.get("similarity_score", 0),
        metrics={m.key: player_row.get(m.key) for m in config.metrics},
        percentiles=_compute_percentiles(player_row, all_profiles, config),
        sample_size=player_row.get(config.count_field, 0),
    )

    # Extract archetype data
    archetype_data = ArchetypeData(
        name=archetype.name.replace("_", " ").title(),
        description=getattr(archetype, "description", ""),
        target_profile=getattr(archetype, "target_profile", {}),
    )

    # Get radar features (subset for readability)
    if position_type == "forward":
        radar_features = ["danger_rate", "central_pct", "avg_separation",
                          "avg_entry_speed", "avg_defensive_line_dist", "quick_break_pct"]
    elif position_type == "defender":
        radar_features = ["stop_danger_rate", "reduce_danger_rate", "pressing_rate",
                          "goal_side_rate", "avg_engagement_distance"]
    else:
        radar_features = ["pass_success_rate", "avg_pass_distance", "long_pass_pct",
                          "quick_distribution_pct"]

    # Filter to available features
    radar_features = [f for f in radar_features if f in player.metrics]

    # Build metric labels
    metric_labels = {m.key: m.label for m in config.metrics}

    report_config = ReportConfig(
        include_radar=True,
        include_metrics=True,
        include_insight=insight is not None,
        radar_features=radar_features,
        metric_labels=metric_labels,
        model_name=model_name,
    )

    report = ScoutingReport(
        player=player,
        archetype=archetype_data,
        config=report_config,
        insight=insight,
        all_profiles=all_profiles,
    )

    return report.generate()


def _compute_percentiles(
    player_row: dict,
    all_profiles: Any,
    config: Any,
) -> dict[str, float]:
    """Compute percentiles for player metrics."""
    import numpy as np

    percentiles = {}
    for metric in config.metrics:
        key = metric.key
        value = player_row.get(key)
        if value is None or key not in all_profiles.columns:
            continue

        all_values = all_profiles[key].drop_nulls().to_numpy()
        if len(all_values) == 0:
            continue

        percentile = (np.sum(all_values <= value) / len(all_values)) * 100
        percentiles[key] = percentile

    return percentiles
