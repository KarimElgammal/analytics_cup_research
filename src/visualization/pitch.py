"""Pitch visualizations using mplsoccer."""

import matplotlib.pyplot as plt
from mplsoccer import Pitch
import polars as pl
import numpy as np

from src.utils.config import PITCH_LENGTH, PITCH_WIDTH


def create_pitch(pitch_type: str = "skillcorner") -> tuple[plt.Figure, plt.Axes]:
    """Create a basic pitch figure."""
    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        pitch_color="#1a472a",
        line_color="white",
        linewidth=1,
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    return fig, ax


def plot_entry_locations(
    entries: pl.DataFrame,
    color_by: str = "is_dangerous",
    title: str = "Final Third Entry Locations"
) -> plt.Figure:
    """Plot entry locations on pitch, colored by danger or zone."""
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        pitch_color="#1a472a",
        line_color="white",
        linewidth=1,
        half=True,
    )
    fig, ax = pitch.draw(figsize=(10, 8))

    x = entries.select("x_end").to_numpy().flatten()
    y = entries.select("y_end").to_numpy().flatten()

    if color_by == "is_dangerous":
        colors = entries.select("is_dangerous").to_numpy().flatten()
        cmap = plt.cm.RdYlGn_r
        scatter = ax.scatter(
            x, y,
            c=colors,
            cmap=cmap,
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Dangerous", shrink=0.6)
    else:
        zone_colors = {
            "central": "#e74c3c",
            "half_space": "#f39c12",
            "wide": "#3498db",
        }
        if "entry_zone" in entries.columns:
            zones = entries.select("entry_zone").to_numpy().flatten()
            colors = [zone_colors.get(z, "#95a5a6") for z in zones]
        else:
            colors = "#95a5a6"

        ax.scatter(
            x, y,
            c=colors,
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a472a")

    return fig


def plot_entry_heatmap(
    entries: pl.DataFrame,
    title: str = "Entry Location Heatmap"
) -> plt.Figure:
    """Plot heatmap of entry locations."""
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=PITCH_LENGTH,
        pitch_width=PITCH_WIDTH,
        pitch_color="#1a472a",
        line_color="white",
        linewidth=1,
        half=True,
    )
    fig, ax = pitch.draw(figsize=(10, 8))

    x = entries.select("x_end").to_numpy().flatten()
    y = entries.select("y_end").to_numpy().flatten()

    pitch.kdeplot(
        x, y,
        ax=ax,
        cmap="Reds",
        fill=True,
        levels=50,
        alpha=0.7,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    fig.patch.set_facecolor("#1a472a")

    return fig


def plot_player_entries(
    entries: pl.DataFrame,
    player_name: str,
    title: str | None = None
) -> plt.Figure:
    """Plot entry locations for a specific player."""
    player_entries = entries.filter(pl.col("player_name") == player_name)

    if title is None:
        title = f"{player_name} - Final Third Entries"

    return plot_entry_locations(player_entries, color_by="is_dangerous", title=title)


def plot_entry_locations_interactive(
    entries: pl.DataFrame,
    title: str = "Final Third Entry Locations (Hover for details)"
):
    """
    Plot entry locations using Plotly for interactive hover.

    Hover shows: player name, team, speed, danger status, zone.
    SkillCorner coordinates: x from -52.5 to 52.5, y from -34 to 34.
    Final third is x > 17.5 (attacking towards x=52.5).
    """
    import plotly.graph_objects as go

    # Prepare data
    df = entries.to_pandas()

    # Create hover text
    hover_text = []
    for _, row in df.iterrows():
        text = f"<b>{row.get('player_name', 'Unknown')}</b><br>"
        if 'team_name' in row:
            text += f"Team: {row['team_name']}<br>"
        if 'entry_zone' in row:
            text += f"Zone: {row['entry_zone']}<br>"
        if 'speed_avg' in row:
            text += f"Speed: {row['speed_avg']:.1f} m/s<br>"
        if 'is_dangerous' in row:
            text += f"Led to shot: {'Yes' if row['is_dangerous'] else 'No'}<br>"
        if 'danger_rate' in row and not np.isnan(row.get('danger_rate', float('nan'))):
            text += f"Player danger rate: {row['danger_rate']:.0f}%"
        hover_text.append(text)

    # Color by danger
    if 'is_dangerous' in df.columns:
        colors = ['#e74c3c' if d else '#3498db' for d in df['is_dangerous']]
    else:
        colors = '#3498db'

    # Create figure
    fig = go.Figure()

    pitch_color = '#1a472a'
    line_color = 'white'

    # SkillCorner coordinates: x=-52.5 to 52.5, y=-34 to 34
    # Final third: x > 17.5, goal at x=52.5
    # Show attacking half: x from 0 to 52.5

    # Pitch outline (attacking half)
    fig.add_shape(type="rect", x0=0, y0=-34, x1=52.5, y1=34,
                  line=dict(color=line_color, width=2), fillcolor=pitch_color)

    # Centre line (at x=0)
    fig.add_shape(type="line", x0=0, y0=-34, x1=0, y1=34,
                  line=dict(color=line_color, width=1))

    # Final third line (at x=17.5)
    fig.add_shape(type="line", x0=17.5, y0=-34, x1=17.5, y1=34,
                  line=dict(color=line_color, width=1, dash="dash"))

    # Penalty area (16.5m from goal line, 40.32m wide = 20.16m each side)
    fig.add_shape(type="rect", x0=52.5-16.5, y0=-20.16, x1=52.5, y1=20.16,
                  line=dict(color=line_color, width=1))

    # Goal area (5.5m from goal line, 18.32m wide = 9.16m each side)
    fig.add_shape(type="rect", x0=52.5-5.5, y0=-9.16, x1=52.5, y1=9.16,
                  line=dict(color=line_color, width=1))

    # Goal (behind the line)
    fig.add_shape(type="rect", x0=52.5, y0=-3.66, x1=54, y1=3.66,
                  line=dict(color=line_color, width=2), fillcolor='white')

    # Penalty spot (11m from goal)
    fig.add_trace(go.Scatter(x=[52.5-11], y=[0], mode='markers',
                             marker=dict(size=6, color=line_color),
                             hoverinfo='skip', showlegend=False))

    # Penalty arc (radius 9.15m from penalty spot, only part outside box)
    arc_angles = np.linspace(-0.6, 0.6, 30)  # radians
    arc_x = 52.5 - 11 + 9.15 * np.cos(arc_angles)
    arc_y = 9.15 * np.sin(arc_angles)
    # Only show part outside penalty area
    mask = arc_x < 36
    fig.add_trace(go.Scatter(x=arc_x[mask], y=arc_y[mask], mode='lines',
                             line=dict(color=line_color, width=1),
                             hoverinfo='skip', showlegend=False))

    # Add entry points
    fig.add_trace(go.Scatter(
        x=df['x_end'],
        y=df['y_end'],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=1, color='white'),
            opacity=0.8,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Entries'
    ))

    # Layout - horizontal pitch (wider than tall)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='white')),
        xaxis=dict(
            range=[-2, 56],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain='domain',
        ),
        yaxis=dict(
            range=[-38, 38],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor=pitch_color,
        paper_bgcolor=pitch_color,
        showlegend=True,
        legend=dict(
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.3)',
            x=1, y=1,
            xanchor='right'
        ),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Add legend items
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#e74c3c'),
        name='Led to shot'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#3498db'),
        name='No shot'
    ))

    return fig
