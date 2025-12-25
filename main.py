"""Finding Alvarez in the A-League - Streamlit App

SkillCorner X PySport Analytics Cup 2026 - Research Track

A player similarity study using broadcast tracking data to identify
A-League players with Julian Alvarez-like characteristics.
"""

import streamlit as st
import polars as pl
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Finding Alvarez in the A-League",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data(show_spinner="Loading tracking data...")
def load_data():
    """Load and process all match data."""
    from src.data.loader import load_all_events, add_team_names
    from src.analysis.entries import detect_entries, classify_entries

    events = load_all_events()
    events = add_team_names(events)
    entries = detect_entries(events)
    entries = classify_entries(entries)

    return entries


@st.cache_data(show_spinner="Building player profiles...")
def build_profiles(entries: pl.DataFrame):
    """Build player profiles from entries."""
    from src.analysis.profiles import build_player_profiles, filter_profiles

    profiles = build_player_profiles(entries)
    profiles = filter_profiles(profiles, min_entries=3)
    return profiles


@st.cache_data(show_spinner="Computing similarity scores...")
def compute_rankings(profiles: pl.DataFrame):
    """Compute similarity rankings."""
    from src.analysis.similarity import compute_similarity_scores, rank_candidates

    ranked = compute_similarity_scores(profiles)
    ranked = rank_candidates(ranked, min_entries=3)
    return ranked


def main():
    st.title("⚽ Finding Alvarez in the A-League")
    st.markdown(
        """
        **SkillCorner X PySport Analytics Cup 2026 - Research Track**

        A player similarity study using broadcast tracking data to identify
        A-League players with Julian Alvarez-like characteristics.
        """
    )

    # Load data
    entries = load_data()
    profiles = build_profiles(entries)
    ranked = compute_rankings(profiles)

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown(
        """
        Julian Alvarez represents a modern forward archetype:
        intelligent movement, spatial awareness, and clinical finishing.

        This analysis uses SkillCorner tracking data to find A-League
        players with similar characteristics.
        """
    )

    st.sidebar.header("Data Summary")
    st.sidebar.metric("Total Entries", len(entries))
    st.sidebar.metric("Matches Analysed", entries["match_id"].n_unique())
    st.sidebar.metric("Players Profiled", len(profiles))

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Rankings",
        "📊 Profile Comparison",
        "🗺️ Entry Locations",
        "📖 Methodology"
    ])

    with tab1:
        render_rankings_tab(ranked, profiles)

    with tab2:
        render_comparison_tab(ranked, profiles)

    with tab3:
        render_locations_tab(entries, ranked)

    with tab4:
        render_methodology_tab()


def render_rankings_tab(ranked: pl.DataFrame, profiles: pl.DataFrame):
    """Render the rankings tab."""
    st.header("Similarity Rankings")
    st.markdown(
        """
        Players ranked by similarity to the Alvarez archetype based on:
        separation, central positioning, danger creation, speed, and link-up play.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        from src.visualization.radar import plot_similarity_ranking

        fig = plot_similarity_ranking(
            ranked,
            top_n=10,
            title="Top 10 Alvarez-Like Players"
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 5 Candidates")
        display_cols = ["rank", "player_name", "team_name", "similarity_score", "total_entries"]
        available = [c for c in display_cols if c in ranked.columns]
        st.dataframe(
            ranked.head(5).select(available).to_pandas(),
            hide_index=True,
            use_container_width=True
        )


def render_comparison_tab(ranked: pl.DataFrame, profiles: pl.DataFrame):
    """Render the profile comparison tab."""
    st.header("Profile Comparison")

    from src.visualization.radar import plot_radar_comparison, normalize_for_radar, RADAR_FEATURES

    features = list(RADAR_FEATURES.keys())

    # Player selector
    player_names = ranked["player_name"].to_list()
    selected = st.multiselect(
        "Select players to compare (max 4)",
        player_names[:20],
        default=player_names[:3],
        max_selections=4
    )

    if selected:
        # Normalize and prepare data
        norm_profiles = normalize_for_radar(ranked, features)

        players_data = []
        for name in selected:
            row = norm_profiles.filter(pl.col("player_name") == name).to_dicts()
            if row:
                row = row[0]
                player_data = {"name": f"{name} ({row.get('team_name', '')})"}
                for f in features:
                    player_data[f] = row.get(f"{f}_norm", 0)
                players_data.append(player_data)

        if players_data:
            fig = plot_radar_comparison(
                players_data,
                features=features,
                title="Player Profile Comparison"
            )
            st.pyplot(fig)
            plt.close()

    # Detailed stats table
    st.subheader("Detailed Statistics")
    if selected:
        detail_cols = ["player_name", "team_name", "total_entries", "danger_rate",
                       "central_pct", "half_space_pct", "avg_separation",
                       "avg_entry_speed", "carry_pct"]
        available = [c for c in detail_cols if c in ranked.columns]
        filtered = ranked.filter(pl.col("player_name").is_in(selected))
        st.dataframe(filtered.select(available).to_pandas(), hide_index=True)


def render_locations_tab(entries: pl.DataFrame, ranked: pl.DataFrame):
    """Render entry locations tab."""
    st.header("Entry Locations")

    from src.visualization.pitch import (
        plot_entry_heatmap,
        plot_entry_locations,
        plot_entry_locations_interactive
    )

    # Player selector
    top_players = ranked.head(10)["player_name"].to_list()
    selected_player = st.selectbox("Select player", ["All Players"] + top_players)

    if selected_player == "All Players":
        player_entries = entries
        title = "All Final Third Entries"
    else:
        player_entries = entries.filter(pl.col("player_name") == selected_player)
        team = entries.filter(pl.col("player_name") == selected_player)["team_name"].first()
        title = f"{selected_player} ({team}) - Final Third Entries"

    # Interactive Plotly chart (hover to see player details)
    st.subheader("Interactive Entry Map")
    st.caption("Hover over points to see player details, zone, speed, and danger status")

    # Add danger_rate to entries for hover display
    player_danger_rates = ranked.select(["player_name", "danger_rate"])
    entries_with_rate = player_entries.join(
        player_danger_rates, on="player_name", how="left"
    )

    plotly_fig = plot_entry_locations_interactive(
        entries_with_rate,
        title=f"{title} (Hover for details)"
    )
    st.plotly_chart(plotly_fig, use_container_width=True)

    # Static charts below
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entry Points (Static)")
        fig = plot_entry_locations(player_entries, color_by="is_dangerous", title=title)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Heat Map")
        if len(player_entries) >= 5:
            fig = plot_entry_heatmap(player_entries, title=f"{title} - Density")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need at least 5 entries for heatmap")


def render_methodology_tab():
    """Render methodology explanation."""
    st.header("Methodology")

    st.markdown(
        """
        ### The Alvarez Archetype

        We derive the Alvarez archetype from **StatsBomb event data** covering
        World Cup and Copa América matches:

        | StatsBomb Metric | Value | Interpretation |
        |------------------|-------|----------------|
        | Shot accuracy | 60% | Clinical finisher |
        | Conversion rate | 20% | High-pressure performer |
        | Box touches | 24 | Comfortable in danger areas |
        | Pass accuracy | 78.9% | Takes calculated risks |
        | Dribble success | 50% | NOT a dribbler |

        These traits are mapped to SkillCorner tracking metrics:

        | StatsBomb Trait | SkillCorner Mapping | Weight |
        |-----------------|---------------------|--------|
        | 20% conversion | `danger_rate` | 20% |
        | Intelligent movement | `avg_separation` | 18% |
        | 24 box touches | `central_pct` + `half_space_pct` | 22% |
        | Link-up play | `avg_passing_options` | 10% |
        | NOT a dribbler | `carry_pct` (low) | 3% |

        ### Feature Engineering

        From SkillCorner's dynamic events, we detect final third entries and compute:

        - **Spatial**: Separation from defenders, defensive line distance
        - **Zone preferences**: Central, half-space, wide percentages
        - **Outcomes**: Danger rate (entries leading to shots)

        ### Similarity Scoring

        We use **weighted cosine similarity** on z-score normalised features.
        The target uses 90th percentile values for positive traits.
        Players with <3 entries excluded.

        ### Limitations

        - **Sample size**: Only 10 matches
        - **Cross-dataset**: Archetype from StatsBomb, candidates from SkillCorner
        - **No position labels**: Filtered by entry count only
        """
    )


if __name__ == "__main__":
    main()
