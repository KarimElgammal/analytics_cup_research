"""Reusable PDF components for report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fpdf import FPDF


@dataclass(frozen=True, slots=True)
class Colors:
    """Color palette for reports."""

    PRIMARY = (41, 128, 185)  # Blue
    SECONDARY = (52, 73, 94)  # Dark gray
    SUCCESS = (39, 174, 96)  # Green
    WARNING = (243, 156, 18)  # Orange
    DANGER = (231, 76, 60)  # Red
    LIGHT = (236, 240, 241)  # Light gray
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    TEXT = (44, 62, 80)  # Dark text


@dataclass(frozen=True, slots=True)
class Fonts:
    """Font configuration."""

    HEADER = ("Helvetica", "B", 24)
    SUBHEADER = ("Helvetica", "B", 16)
    SECTION = ("Helvetica", "B", 12)
    BODY = ("Helvetica", "", 10)
    SMALL = ("Helvetica", "", 8)
    METRIC_LABEL = ("Helvetica", "B", 9)
    METRIC_VALUE = ("Helvetica", "", 9)


class PDFComponent:
    """Base class for PDF components."""

    def __init__(self, pdf: "FPDF"):
        self.pdf = pdf

    def set_font(self, font_tuple: tuple[str, str, int]) -> None:
        """Set font from a tuple."""
        self.pdf.set_font(*font_tuple)

    def set_color(self, color: tuple[int, int, int], fill: bool = False) -> None:
        """Set text or fill color."""
        if fill:
            self.pdf.set_fill_color(*color)
        else:
            self.pdf.set_text_color(*color)


class HeaderComponent(PDFComponent):
    """Report header with player info."""

    def render(
        self,
        player_name: str,
        team: str,
        age: int | None,
        position: str,
        archetype: str,
        similarity: float,
    ) -> None:
        """Render the header section."""
        # Title bar
        self.set_color(Colors.PRIMARY, fill=True)
        self.pdf.rect(0, 0, 210, 35, "F")

        # Player name
        self.set_font(Fonts.HEADER)
        self.set_color(Colors.WHITE)
        self.pdf.set_xy(10, 8)
        self.pdf.cell(0, 10, player_name, align="L")

        # Team and age
        self.set_font(Fonts.BODY)
        age_str = f"Age {age}" if age else "Age unknown"
        self.pdf.set_xy(10, 20)
        self.pdf.cell(0, 6, f"{team} | {age_str} | {position}", align="L")

        # Similarity badge
        self._render_similarity_badge(similarity, archetype)

        self.pdf.ln(25)

    def _render_similarity_badge(self, similarity: float, archetype: str) -> None:
        """Render similarity score badge."""
        badge_x = 150
        badge_y = 8

        # Badge background
        if similarity >= 90:
            color = Colors.SUCCESS
        elif similarity >= 75:
            color = Colors.PRIMARY
        elif similarity >= 60:
            color = Colors.WARNING
        else:
            color = Colors.DANGER

        self.set_color(color, fill=True)
        self.pdf.rect(badge_x, badge_y, 50, 20, "F")

        # Similarity text
        self.set_font(("Helvetica", "B", 18))
        self.set_color(Colors.WHITE)
        self.pdf.set_xy(badge_x, badge_y + 2)
        self.pdf.cell(50, 10, f"{similarity:.1f}%", align="C")

        # Archetype name
        self.set_font(Fonts.SMALL)
        self.pdf.set_xy(badge_x, badge_y + 12)
        self.pdf.cell(50, 6, f"vs {archetype}", align="C")


class MetricsTableComponent(PDFComponent):
    """Metrics table with percentile bars."""

    def render(
        self,
        metrics: dict[str, float],
        percentiles: dict[str, float],
        metric_labels: dict[str, str],
    ) -> None:
        """Render metrics table with percentile visualization."""
        self._render_section_header("Player Metrics")

        # Table header
        self.set_color(Colors.SECONDARY, fill=True)
        self.pdf.set_fill_color(*Colors.LIGHT)

        col_widths = [55, 25, 25, 85]  # Metric, Value, Pctl, Bar

        self.set_font(Fonts.SECTION)
        self.set_color(Colors.TEXT)
        self.pdf.cell(col_widths[0], 8, "Metric", border=1, fill=True)
        self.pdf.cell(col_widths[1], 8, "Value", border=1, align="C", fill=True)
        self.pdf.cell(col_widths[2], 8, "Pctl", border=1, align="C", fill=True)
        self.pdf.cell(col_widths[3], 8, "Percentile Rank", border=1, align="C", fill=True)
        self.pdf.ln()

        # Table rows
        self.set_font(Fonts.METRIC_VALUE)
        for key, value in metrics.items():
            if value is None:
                continue

            label = metric_labels.get(key, key)
            pctl = percentiles.get(key)

            self.set_color(Colors.TEXT)
            self.pdf.cell(col_widths[0], 7, label[:25], border=1)
            self.pdf.cell(col_widths[1], 7, f"{value:.1f}", border=1, align="C")

            # Handle None percentile
            if pctl is not None:
                self.pdf.cell(col_widths[2], 7, f"P{pctl:.0f}", border=1, align="C")
                self._render_percentile_bar(col_widths[3], pctl)
            else:
                self.pdf.cell(col_widths[2], 7, "N/A", border=1, align="C")
                self.pdf.cell(col_widths[3], 7, "", border=1)

            self.pdf.ln()

        self.pdf.ln(5)

    def _render_percentile_bar(self, width: float, percentile: float) -> None:
        """Render a percentile bar inside a cell."""
        x = self.pdf.get_x()
        y = self.pdf.get_y()

        # Cell border
        self.pdf.cell(width, 7, "", border=1)

        # Bar background
        bar_margin = 2
        bar_width = width - 2 * bar_margin
        bar_height = 5

        self.set_color(Colors.LIGHT, fill=True)
        self.pdf.rect(x + bar_margin, y + 1, bar_width, bar_height, "F")

        # Filled portion
        fill_width = (percentile / 100) * bar_width
        if percentile >= 75:
            color = Colors.SUCCESS
        elif percentile >= 50:
            color = Colors.PRIMARY
        elif percentile >= 25:
            color = Colors.WARNING
        else:
            color = Colors.DANGER

        self.set_color(color, fill=True)
        self.pdf.rect(x + bar_margin, y + 1, fill_width, bar_height, "F")

    def _render_section_header(self, title: str) -> None:
        """Render a section header."""
        self.set_font(Fonts.SUBHEADER)
        self.set_color(Colors.SECONDARY)
        self.pdf.cell(0, 10, title, ln=True)


class InsightComponent(PDFComponent):
    """AI-generated insight section."""

    def render(self, insight: str, model_name: str | None = None) -> None:
        """Render the AI insight section."""
        self._render_section_header("Scouting Insight")

        # Clean markdown from insight
        cleaned_insight = self._clean_markdown(insight)

        self.set_font(Fonts.BODY)
        self.set_color(Colors.TEXT)

        # Multi-line text (no border box - cleaner look)
        self.pdf.set_x(10)
        self.pdf.multi_cell(190, 5, cleaned_insight)

        # Model attribution
        if model_name:
            self.pdf.ln(2)
            self.set_font(Fonts.SMALL)
            self.set_color(Colors.SECONDARY)
            self.pdf.cell(0, 5, f"Generated by {model_name}", align="R")

        self.pdf.ln(5)

    def _clean_markdown(self, text: str) -> str:
        """Convert markdown to plain text suitable for PDF."""
        import re

        # Convert headers to uppercase with newlines
        text = re.sub(r'#{1,6}\s*\*\*(.+?)\*\*', r'\n\1\n', text)  # ### **Header**
        text = re.sub(r'#{1,6}\s*(.+)', r'\n\1\n', text)  # ### Header

        # Remove bold/italic markers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # *italic*
        text = re.sub(r'__(.+?)__', r'\1', text)  # __bold__
        text = re.sub(r'_(.+?)_', r'\1', text)  # _italic_

        # Remove bullet points but keep text
        text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)

        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _render_section_header(self, title: str) -> None:
        """Render a section header."""
        self.set_font(Fonts.SUBHEADER)
        self.set_color(Colors.SECONDARY)
        self.pdf.cell(0, 10, title, ln=True)


class FooterComponent(PDFComponent):
    """Report footer."""

    def render(self, page_num: int, total_pages: int) -> None:
        """Render the footer."""
        self.pdf.set_y(-15)
        self.set_font(Fonts.SMALL)
        self.set_color(Colors.SECONDARY)

        # Left: source
        self.pdf.set_x(10)
        self.pdf.cell(60, 10, "SkillCorner X PySport Analytics Cup 2026", align="L")

        # Center: page number
        self.pdf.cell(70, 10, f"Page {page_num}/{total_pages}", align="C")

        # Right: date
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        self.pdf.cell(60, 10, date_str, align="R")


class RadarChartComponent(PDFComponent):
    """Radar chart image component."""

    def render(self, image_path: str, width: float = 100) -> None:
        """Render radar chart image."""
        self._render_section_header("Profile Comparison")

        # Center the image
        x = (210 - width) / 2
        self.pdf.image(image_path, x=x, w=width)
        self.pdf.ln(5)

    def _render_section_header(self, title: str) -> None:
        """Render a section header."""
        self.set_font(Fonts.SUBHEADER)
        self.set_color(Colors.SECONDARY)
        self.pdf.cell(0, 10, title, ln=True)
