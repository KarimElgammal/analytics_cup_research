"""Interactive Archetype Comparison Tool.

A Streamlit app for comparing A-League players against different player archetypes.
Supports forwards (final third entries), defenders (defensive engagements),
and goalkeepers (distribution).

Run: streamlit run app.py
"""

import warnings
import streamlit as st
import polars as pl
import pandas as pd
from packaging import version

# Streamlit version compatibility for width parameter
_ST_VERSION = version.parse(st.__version__)
_USE_NEW_WIDTH = _ST_VERSION >= version.parse("1.41.0")


def get_dataframe_width_kwargs():
    """Return appropriate width kwargs for st.dataframe based on Streamlit version."""
    if _USE_NEW_WIDTH:
        return {"width": "stretch"}
    return {"use_container_width": True}


# Core imports
from src.data.loader import load_all_events, add_team_names
from src.data.player_ages import add_ages_to_profiles
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles
from src.analysis.defenders import detect_defensive_actions, build_defender_profiles, filter_defender_profiles
from src.analysis.goalkeepers import detect_gk_actions, build_goalkeeper_profiles, filter_goalkeeper_profiles
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine
from src.visualization.radar import (
    plot_radar_comparison,
    plot_similarity_ranking,
    normalize_for_radar_percentile,
)
from src.utils.ai_insights import (
    generate_similarity_insight,
    has_valid_token,
    get_available_models,
    get_default_model,
)
from src.utils.rate_limiter import (
    check_daily_limit,
    check_backend_budget,
    increment_call,
    get_usage_stats,
    get_backend_from_model,
    is_localhost,
    MAX_CALLS_PER_SESSION,
)

# Archetype imports
from src.archetypes.base import (
    DEFENDER_FEATURE_NAMES,
    GOALKEEPER_FEATURE_NAMES,
)
from src.archetypes.forwards import FORWARD_ARCHETYPE_OPTIONS
from src.archetypes.defenders import (
    DEFENDER_ARCHETYPES,
    DEFENDER_WEIGHTS,
    DEFENDER_DIRECTIONS,
)
from src.archetypes.goalkeepers import GOALKEEPER_ARCHETYPES

# Page config
st.set_page_config(
    page_title="Finding Alvarez in the A-League",
    page_icon="\u26BD",
    layout="wide",
)

# Sidebar links
st.sidebar.markdown("### Links")
st.sidebar.markdown("[Documentation](https://karimelgammal.github.io/analytics_cup_research/) | [GitHub](https://github.com/KarimElgammal/analytics_cup_research)")
st.sidebar.markdown("[Glossary](https://karimelgammal.github.io/analytics_cup_research/glossary/) - Metric definitions")
st.sidebar.markdown("---")

st.title("\u26BD Finding Alvarez (and Others) in the A-League")

# Position selector at the top
position = st.radio(
    "Select Position Type",
    ["Forwards", "Defenders", "Goalkeepers"],
    horizontal=True,
    help="Each position uses different event data and metrics."
)

st.markdown("---")


# --- Data Loading (cached) ---

@st.cache_data(ttl=7776000, show_spinner="Loading A-League data...")
def load_forward_data():
    """Load and prepare forward profiles from final third entries."""
    events = load_all_events()
    events = add_team_names(events)
    entries = detect_entries(events)
    entries = classify_entries(entries)
    profiles = build_player_profiles(entries)
    profiles = filter_profiles(profiles, min_entries=3)
    return profiles, len(entries), events["match_id"].n_unique()


@st.cache_data(ttl=7776000, show_spinner="Loading A-League defensive data...")
def load_defender_data():
    """Load and prepare defender profiles from defensive engagements."""
    events = load_all_events()
    events = add_team_names(events)
    actions = detect_defensive_actions(events)
    profiles = build_defender_profiles(actions)
    profiles = filter_defender_profiles(profiles, min_engagements=5)
    return profiles, len(actions), events["match_id"].n_unique()


@st.cache_data(ttl=7776000, show_spinner="Loading A-League goalkeeper data...")
def load_goalkeeper_data():
    """Load and prepare goalkeeper profiles from distribution events."""
    events = load_all_events()
    events = add_team_names(events)
    actions = detect_gk_actions(events)
    profiles = build_goalkeeper_profiles(actions)
    profiles = filter_goalkeeper_profiles(profiles, min_distributions=10)
    return profiles, len(actions), events["match_id"].n_unique()


@st.cache_data(ttl=7776000, show_spinner="Loading archetype...")
def load_forward_archetype(player_key: str):
    """Load forward archetype from StatsBomb data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Archetype.from_statsbomb(player_key, verbose=False)


# --- Position-specific Configuration ---

DEFENDER_OPTIONS = [
    ("Gvardiol (CRO) - Ball-playing CB", "gvardiol"),
    ("Van Dijk (NED) - Commanding CB", "vandijk"),
    ("Hakimi (MAR) - Attacking Wing-back", "hakimi"),
]

GOALKEEPER_OPTIONS = [
    ("Neuer (GER) - Modern Sweeper", "neuer"),
    ("Lloris (FRA) - Direct Distributor", "lloris"),
    ("Bounou (MAR) - Balanced Long", "bounou"),
]


def get_defender_archetype(key: str) -> Archetype:
    """Get defender archetype from pre-computed configs."""
    config = DEFENDER_ARCHETYPES[key]
    return config.to_archetype(DEFENDER_WEIGHTS, DEFENDER_DIRECTIONS, DEFENDER_FEATURE_NAMES)


def get_goalkeeper_archetype(key: str) -> Archetype:
    """Get goalkeeper archetype with style-specific weights."""
    config, weights, directions = GOALKEEPER_ARCHETYPES[key]
    return config.to_archetype(weights, directions, GOALKEEPER_FEATURE_NAMES)


# --- Main Logic ---

if position == "Forwards":
    profiles, n_events, n_matches = load_forward_data()
    event_type = "entries"
    position_type = "forward"

    st.sidebar.header("Forward Archetype")
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in FORWARD_ARCHETYPE_OPTIONS],
        index=0,
    )
    selected_key = next(key for label, key in FORWARD_ARCHETYPE_OPTIONS if label == selected_label)
    archetype = load_forward_archetype(selected_key)

    display_cols = [
        "rank", "player_name", "age", "team_name", "similarity_score",
        "total_entries", "danger_rate", "avg_separation"
    ]
    col_names = ["Rank", "Player", "Age", "Team", "Similarity %", "Entries", "Danger %", "Separation (m)"]
    radar_features = ["danger_rate", "central_pct", "avg_separation", "avg_entry_speed"]
    caption = "Forward archetypes computed from StatsBomb World Cup 2022 event data."

elif position == "Defenders":
    profiles, n_events, n_matches = load_defender_data()
    event_type = "engagements"
    position_type = "defender"

    st.sidebar.header("Defender Archetype")
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in DEFENDER_OPTIONS],
        index=0,
    )
    selected_key = next(key for label, key in DEFENDER_OPTIONS if label == selected_label)
    archetype = get_defender_archetype(selected_key)

    display_cols = [
        "rank", "player_name", "age", "team_name", "similarity_score",
        "total_engagements", "stop_danger_rate", "pressing_rate"
    ]
    col_names = ["Rank", "Player", "Age", "Team", "Similarity %", "Engagements", "Stop Danger %", "Pressing %"]
    radar_features = ["stop_danger_rate", "reduce_danger_rate", "pressing_rate", "goal_side_rate", "avg_engagement_distance"]
    caption = "Archetypes computed from StatsBomb open data (World Cup 2022, Euro 2024)."

else:  # Goalkeepers
    profiles, n_events, n_matches = load_goalkeeper_data()
    event_type = "distributions"
    position_type = "goalkeeper"

    st.sidebar.header("Goalkeeper Archetype")
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in GOALKEEPER_OPTIONS],
        index=0,
    )
    selected_key = next(key for label, key in GOALKEEPER_OPTIONS if label == selected_label)
    archetype = get_goalkeeper_archetype(selected_key)

    display_cols = [
        "rank", "player_name", "age", "team_name", "similarity_score",
        "total_distributions", "pass_success_rate", "long_pass_pct"
    ]
    col_names = ["Rank", "Player", "Age", "Team", "Similarity %", "Distributions", "Pass Success %", "Long Pass %"]
    radar_features = ["pass_success_rate", "avg_pass_distance", "long_pass_pct", "quick_distribution_pct"]
    caption = "All archetypes computed from StatsBomb open data (Euro 2024 / World Cup 2022)."

# Sidebar controls
top_n = st.sidebar.slider("Number of results", min_value=5, max_value=min(20, len(profiles)), value=min(10, len(profiles)))

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Stats")
st.sidebar.markdown(f"**Events:** {n_events:,} {event_type}")
st.sidebar.markdown(f"**Matches:** {n_matches}")
st.sidebar.markdown(f"**Players:** {len(profiles)} qualified")

st.sidebar.markdown("---")
st.sidebar.markdown("### Methodology")
st.sidebar.markdown("Weighted cosine similarity on z-score normalised features.")

# Run similarity engine
engine = SimilarityEngine(archetype)
engine.fit(profiles)
results = engine.rank(top_n=top_n)

# Add ages to results
results = add_ages_to_profiles(results)

# Archetype info panel
st.subheader(f"Finding {archetype.name.replace('_', ' ').title()}-like Players")
with st.expander("Archetype Profile", expanded=False):
    st.markdown(archetype.description)
    st.markdown("**Feature Weights:**")
    importances = engine.get_feature_importance()
    for item in importances[:5]:
        pct = item["weight"] * 100
        st.markdown(f"- `{item['feature']}`: {pct:.0f}%")

# View selector
view_options = ["Rankings", "Radar Profile", "AI Insights"]
if "selected_view" not in st.session_state:
    st.session_state.selected_view = "Rankings"

selected_view = st.radio(
    "View",
    view_options,
    index=view_options.index(st.session_state.selected_view),
    horizontal=True,
    key="view_selector",
)
st.session_state.selected_view = selected_view

st.markdown("---")

if selected_view == "Rankings":
    st.subheader(f"Top {top_n} A-League {position}")

    available_cols = [c for c in display_cols if c in results.columns]
    df_display = results.select(available_cols).to_pandas()
    df_display.columns = col_names[:len(available_cols)]

    format_dict = {"Similarity %": "{:.1f}"}
    for col in df_display.columns:
        if "%" in col and col != "Similarity %":
            format_dict[col] = "{:.1f}"
        elif "(m)" in col:
            format_dict[col] = "{:.2f}"
        elif col == "Age":
            format_dict[col] = lambda x: f"{int(x)}" if pd.notna(x) else ""

    st.dataframe(
        df_display.style.format(format_dict).background_gradient(
            subset=["Similarity %"],
            cmap="Greens",
        ),
        hide_index=True,
        **get_dataframe_width_kwargs(),
    )

    # Bar chart visualization
    st.markdown("#### Similarity Rankings")
    fig_bar = plot_similarity_ranking(
        results,
        top_n=min(top_n, 10),
        title=f"Top {min(top_n, 10)} {archetype.name.replace('_', ' ').title()}-like {position}",
    )
    st.pyplot(fig_bar)

elif selected_view == "Radar Profile":
    st.subheader("Radar Profile Comparison")

    # Prepare data for radar chart - use percentile ranks against full dataset
    top_3 = results.head(3)
    normalized = normalize_for_radar_percentile(top_3, profiles, radar_features)

    # Build player dicts for radar
    players_for_radar = []
    for row in normalized.to_dicts():
        player_dict = {"name": row.get("player_name", "Unknown")}
        for f in radar_features:
            norm_key = f"{f}_norm"
            player_dict[f] = row.get(norm_key, row.get(f, 50))
        players_for_radar.append(player_dict)

    # Get target profile from archetype
    target_profile = {}
    if hasattr(archetype, 'target_profile'):
        target_profile = archetype.target_profile
    elif hasattr(archetype, 'features'):
        for f_name, f_data in archetype.features.items():
            if f_name in radar_features:
                target_profile[f_name] = f_data.get("target", 50)

    fig_radar = plot_radar_comparison(
        players_for_radar,
        features=radar_features,
        title=f"Top 3 vs {archetype.name.replace('_', ' ').title()} Target",
        include_alvarez=True,
        target_profile=target_profile,
        target_name=archetype.name.replace("_", " ").title(),
    )
    st.pyplot(fig_radar)

    if position_type == "forward":
        st.caption("Player values are percentile ranks (0-100) from SkillCorner tracking data. *Target value is guessed (no StatsBomb equivalent).")
    else:
        st.caption("Player values are percentile ranks (0-100) from SkillCorner tracking data. Target values computed from StatsBomb World Cup 2022 data.")

elif selected_view == "AI Insights":
    st.subheader("AI Scouting Insight")

    if has_valid_token():
        # init session state
        if "ai_calls" not in st.session_state:
            st.session_state.ai_calls = 0
        if "last_insight" not in st.session_state:
            st.session_state.last_insight = None
        if "last_archetype" not in st.session_state:
            st.session_state.last_archetype = None

        # check limits (bypass on localhost)
        daily_ok, daily_remaining = check_daily_limit()
        session_ok = is_localhost() or st.session_state.ai_calls < MAX_CALLS_PER_SESSION

        # Model selector
        available_models = get_available_models()
        if available_models:
            model_options = {display: key for key, display in available_models}
            default_model = get_default_model()
            default_display = next(
                (display for key, display in available_models if key == default_model),
                available_models[0][1]
            )
            selected_display = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                index=list(model_options.keys()).index(default_display) if default_display in model_options else 0,
            )
            selected_model = model_options[selected_display]
        else:
            selected_model = get_default_model()

        # Check backend-specific budget
        backend = get_backend_from_model(selected_model)
        budget_ok, budget_remaining = check_backend_budget(backend)

        if not daily_ok:
            st.warning("Daily AI insight limit reached. Try again tomorrow.")
        elif not budget_ok:
            st.warning(f"{backend.title()} monthly budget exhausted. Try the other backend.")
        elif not session_ok:
            st.info(f"You've used {MAX_CALLS_PER_SESSION} AI insights this session. Refresh to reset.")
        else:
            if st.button("Generate AI Insight", type="primary"):
                with st.spinner("Generating AI scouting insight..."):
                    insight = generate_similarity_insight(
                        results,
                        archetype,
                        top_n=5,
                        model=selected_model,
                        position_type=position_type,
                    )
                    st.session_state.ai_calls += 1
                    st.session_state.last_insight = insight
                    st.session_state.last_archetype = archetype.name
                    increment_call(backend=backend)

        # Display last insight if it matches current archetype
        if st.session_state.last_insight and st.session_state.last_archetype == archetype.name:
            st.markdown(st.session_state.last_insight)

        # Show usage stats
        stats = get_usage_stats()
        if stats.get("localhost"):
            st.caption("Running locally - no rate limits")
        else:
            st.caption(
                f"Session: {st.session_state.ai_calls}/{MAX_CALLS_PER_SESSION} | "
                f"Today: {stats['daily_calls']}/{stats['daily_limit']} | "
                f"GitHub: ${stats['github_cost']:.2f}/$2 | "
                f"HF: ${stats['huggingface_cost']:.2f}/$2"
            )
    else:
        st.info("Add a token to enable AI-powered scouting insights.")
        st.markdown("""
        **How to enable AI insights:**

        Option 1 (HuggingFace):
        - Save HF token to `hf_token.txt`

        Option 2 (GitHub Models):
        - Save GitHub token to `github_token.txt`

        The AI will generate position-aware scouting recommendations comparing the top candidates to the selected archetype.
        """)

st.markdown("---")
st.caption(caption + " Similarity uses weighted cosine similarity on z-score normalised features.")
