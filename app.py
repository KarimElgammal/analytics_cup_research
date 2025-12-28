"""Interactive Archetype Comparison Tool.

A Streamlit app for comparing A-League players against different player archetypes.
Supports forwards (final third entries), defenders (defensive engagements),
and goalkeepers (distribution).

Run: streamlit run archetype_compare.py
"""

import warnings
import streamlit as st
import polars as pl
from packaging import version

# Streamlit version compatibility for width parameter
# use_container_width deprecated in 1.41+, removed after 2025-12-31
_ST_VERSION = version.parse(st.__version__)
_USE_NEW_WIDTH = _ST_VERSION >= version.parse("1.41.0")

def get_dataframe_width_kwargs():
    """Return appropriate width kwargs for st.dataframe based on Streamlit version."""
    if _USE_NEW_WIDTH:
        return {"width": "stretch"}
    return {"use_container_width": True}

from src.data.loader import load_all_events, add_team_names
from src.data.player_ages import add_ages_to_profiles
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles
from src.analysis.defenders import detect_defensive_actions, build_defender_profiles, filter_defender_profiles
from src.analysis.goalkeepers import detect_gk_actions, build_goalkeeper_profiles, filter_goalkeeper_profiles
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine
from src.core.archetype_factory import ArchetypeFactory
from src.visualization.radar import (
    plot_radar_comparison,
    plot_similarity_ranking,
    normalize_for_radar_percentile,
)
from src.utils.ai_insights import (
    generate_similarity_insight,
    has_valid_token,
    get_available_backend,
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

# Forward archetypes (from StatsBomb)
FORWARD_ARCHETYPES = [
    ("Alvarez (ARG) - Movement-focused", "alvarez"),
    ("Giroud (FRA) - Target man", "giroud"),
    ("Kane (ENG) - Complete forward", "kane"),
    ("Lewandowski (POL) - Clinical poacher", "lewandowski"),
    ("Rashford (ENG) - Pace/dribbling", "rashford"),
    ("En-Nesyri (MAR) - Physical forward", "en_nesyri"),
]

# Cached archetype loading from StatsBomb
@st.cache_resource(show_spinner="Loading archetypes from StatsBomb...")
def get_archetype_factory():
    """Get cached archetype factory."""
    return ArchetypeFactory(verbose=False)

@st.cache_resource(show_spinner="Building defender archetype from World Cup data...")
def load_defender_archetype(player_key: str):
    """Load defender archetype from StatsBomb, with fallback."""
    try:
        factory = get_archetype_factory()
        return factory.build_defender(player_key)
    except Exception as e:
        st.warning(f"Could not load {player_key} from StatsBomb: {e}")
        return None

@st.cache_resource(show_spinner="Building goalkeeper archetype from World Cup data...")
def load_goalkeeper_archetype(player_key: str):
    """Load goalkeeper archetype from StatsBomb, with fallback."""
    try:
        factory = get_archetype_factory()
        return factory.build_goalkeeper(player_key)
    except Exception as e:
        st.warning(f"Could not load {player_key} from StatsBomb: {e}")
        return None

# Defender weights - derived from one-time ML analysis (GradientBoosting, AUC 0.845)
# Weights are STATIC - extracted from feature importances, not computed live
# Used as fallback when StatsBomb API fails
DEFENDER_WEIGHTS = {
    "stop_danger_rate": 0.30,
    "avg_engagement_distance": 0.17,
    "reduce_danger_rate": 0.15,
    "beaten_by_possession_rate": 0.15,
    "beaten_by_movement_rate": 0.08,
    "force_backward_rate": 0.08,
    "pressing_rate": 0.04,
    "goal_side_rate": 0.03,
}

DEFENDER_DIRECTIONS = {
    "stop_danger_rate": 1,
    "reduce_danger_rate": 1,
    "force_backward_rate": 1,
    "beaten_by_possession_rate": -1,
    "beaten_by_movement_rate": -1,
    "pressing_rate": 1,
    "goal_side_rate": 1,
    "avg_engagement_distance": -1,
    "defensive_third_pct": 1,
    "middle_third_pct": 0,
    "attacking_third_pct": -1,
}

# Goalkeeper weights - derived from one-time ML analysis (GradientBoosting, AUC 0.993)
# Weights are STATIC - balanced for style comparison (ML found pass_distance dominates at 98.6%)
# Used as fallback when StatsBomb API fails
GOALKEEPER_WEIGHTS = {
    "pass_success_rate": 0.20,
    "avg_pass_distance": 0.20,
    "long_pass_pct": 0.15,
    "short_pass_pct": 0.15,
    "quick_distribution_pct": 0.10,
    "to_middle_third_pct": 0.10,
    "avg_passing_options": 0.10,
}

GOALKEEPER_DIRECTIONS = {
    "pass_success_rate": 1,
    "long_pass_pct": 1,  # Depends on style
    "short_pass_pct": 1,  # Depends on style
    "avg_pass_distance": 1,  # Depends on style
    "quick_distribution_pct": 1,
    "high_pass_pct": 1,
    "to_middle_third_pct": 1,
    "to_attacking_third_pct": 1,
    "avg_passing_options": 1,
    "hand_pass_pct": -1,  # Less hand passes = more with feet
}


def format_target_table(profile: dict, feature_names: dict[str, str], computed_features: set[str] | None = None) -> str:
    """Format target profile as markdown table."""
    lines = ["Target profile (percentile targets 0-100):", "| Feature | Target | Source |", "|---------|--------|--------|"]
    for key, value in profile.items():
        display_name = feature_names.get(key, key.replace("_", " ").title())
        source = "StatsBomb" if computed_features and key in computed_features else "Estimated"
        lines.append(f"| {display_name} | {value:.0f} | {source} |")
    return "\n".join(lines)


DEFENDER_FEATURE_NAMES = {
    "stop_danger_rate": "Stop Danger %",
    "reduce_danger_rate": "Reduce Danger %",
    "force_backward_rate": "Force Back %",
    "beaten_by_possession_rate": "Beaten (Ball) %",
    "beaten_by_movement_rate": "Beaten (Move) %",
    "pressing_rate": "Pressing %",
    "goal_side_rate": "Goal Side %",
    "avg_engagement_distance": "Engagement Dist",
    "defensive_third_pct": "Defensive 3rd %",
    "middle_third_pct": "Middle 3rd %",
    "attacking_third_pct": "Attacking 3rd %",
}


GK_FEATURE_NAMES = {
    "pass_success_rate": "Pass Success %",
    "short_pass_pct": "Short Pass %",
    "long_pass_pct": "Long Pass %",
    "avg_pass_distance": "Pass Distance",
    "quick_distribution_pct": "Quick Dist %",
    "high_pass_pct": "High Pass %",
    "to_middle_third_pct": "To Middle 3rd %",
    "to_attacking_third_pct": "To Attack 3rd %",
    "avg_passing_options": "Passing Options",
}


# Features computed from StatsBomb (vs estimated)
DEFENDER_COMPUTED = {"stop_danger_rate", "reduce_danger_rate", "pressing_rate", "goal_side_rate", "beaten_by_movement_rate", "avg_engagement_distance"}
GOALKEEPER_COMPUTED = {"pass_success_rate", "long_pass_pct", "avg_pass_distance"}


def create_defender_archetype(name: str, description: str, profile: dict) -> Archetype:
    """Create a defender archetype."""
    full_desc = description + "\n\n" + format_target_table(profile, DEFENDER_FEATURE_NAMES, DEFENDER_COMPUTED)
    return Archetype(
        name=name,
        description=full_desc,
        target_profile=profile,
        weights=DEFENDER_WEIGHTS,
        directions=DEFENDER_DIRECTIONS,
    )


def create_goalkeeper_archetype(name: str, description: str, profile: dict, weights: dict, directions: dict) -> Archetype:
    """Create a goalkeeper archetype."""
    full_desc = description + "\n\n" + format_target_table(profile, GK_FEATURE_NAMES, GOALKEEPER_COMPUTED)
    return Archetype(
        name=name,
        description=full_desc,
        target_profile=profile,
        weights=weights,
        directions=directions,
    )


# Defender archetypes - computed from StatsBomb open data (World Cup 2022, Euro 2024)
# Source: https://github.com/statsbomb/open-data
# Script: scripts/compute_archetype_profiles.py
DEFENDER_ARCHETYPES = {
    "gvardiol": create_defender_archetype(
        "gvardiol",
        "Josko Gvardiol (CRO) - Ball-playing CB\n\nComputed from 3 World Cup 2022 matches (48 events). "
        "Balanced pressing, strong in duels, deeper engagement position.",
        {
            # Computed from StatsBomb open data
            "stop_danger_rate": 67,          # 67% duel success
            "reduce_danger_rate": 32,         # Interceptions + recoveries
            "force_backward_rate": 50,        # Estimated
            "beaten_by_possession_rate": 20,  # Estimated
            "beaten_by_movement_rate": 33,    # 100 - stop_danger_rate
            "pressing_rate": 50,              # Moderate pressing
            "goal_side_rate": 62,             # Good positioning
            "avg_engagement_distance": 32,    # Plays deeper
            "middle_third_pct": 40,           # Estimated
        }
    ),
    "vandijk": create_defender_archetype(
        "vandijk",
        "Virgil van Dijk (NED) - Commanding CB\n\nComputed from 3 Euro 2024 matches (42 events). "
        "Excellent positional defender who reads the game. Minimal duels - wins through positioning.",
        {
            # Computed from StatsBomb open data
            "stop_danger_rate": 50,           # Low duel count, but excellent positioning
            "reduce_danger_rate": 17,         # Rare interceptions (doesn't need to)
            "force_backward_rate": 60,        # Estimated
            "beaten_by_possession_rate": 15,  # Estimated (rarely beaten)
            "beaten_by_movement_rate": 20,    # Estimated (rarely beaten)
            "pressing_rate": 36,              # Lower pressing - reads the game
            "goal_side_rate": 90,             # Excellent positioning (blocks + clearances)
            "avg_engagement_distance": 32,    # Plays deeper
            "defensive_third_pct": 60,        # Estimated
        }
    ),
    "hakimi": create_defender_archetype(
        "hakimi",
        "Achraf Hakimi (MAR) - Attacking Wing-back\n\nComputed from 6 World Cup 2022 matches (148 events). "
        "High pressing, aggressive engagement, plays higher up the pitch.",
        {
            # Computed from StatsBomb open data
            "stop_danger_rate": 41,           # 41% duel success (wing-back position)
            "reduce_danger_rate": 25,         # Interceptions + recoveries
            "force_backward_rate": 45,        # Estimated
            "beaten_by_possession_rate": 30,  # Estimated (takes risks)
            "beaten_by_movement_rate": 59,    # 100 - stop_danger_rate
            "pressing_rate": 59,              # High pressing
            "goal_side_rate": 42,             # Lower (attacking position)
            "avg_engagement_distance": 35,    # Plays higher
            "middle_third_pct": 45,           # Estimated
            "attacking_third_pct": 25,        # Estimated
        }
    ),
}

# Goalkeeper archetypes - different distribution styles
LLORIS_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
LLORIS_WEIGHTS["short_pass_pct"] = 0.20  # Emphasis on playing out
LLORIS_WEIGHTS["long_pass_pct"] = 0.05

LLORIS_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()
LLORIS_DIRECTIONS["short_pass_pct"] = 1
LLORIS_DIRECTIONS["long_pass_pct"] = -1  # Prefers short

LIVAKOVIC_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
LIVAKOVIC_WEIGHTS["long_pass_pct"] = 0.20  # Emphasis on long distribution
LIVAKOVIC_WEIGHTS["short_pass_pct"] = 0.05

LIVAKOVIC_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()
LIVAKOVIC_DIRECTIONS["long_pass_pct"] = 1
LIVAKOVIC_DIRECTIONS["short_pass_pct"] = -1  # Prefers long

# Bounou - Long distribution specialist (39m avg, 42% high balls)
BOUNOU_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
BOUNOU_WEIGHTS["long_pass_pct"] = 0.20
BOUNOU_WEIGHTS["avg_pass_distance"] = 0.20
BOUNOU_WEIGHTS["high_pass_pct"] = 0.15
BOUNOU_WEIGHTS["short_pass_pct"] = 0.00  # Doesn't rely on short passes

BOUNOU_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()
BOUNOU_DIRECTIONS["long_pass_pct"] = 1
BOUNOU_DIRECTIONS["avg_pass_distance"] = 1
BOUNOU_DIRECTIONS["high_pass_pct"] = 1
BOUNOU_DIRECTIONS["short_pass_pct"] = -1

# Neuer weights - balanced distribution (computed from Euro 2024)
NEUER_WEIGHTS = GOALKEEPER_WEIGHTS.copy()
NEUER_WEIGHTS["pass_success_rate"] = 0.25  # Excellent accuracy
NEUER_WEIGHTS["long_pass_pct"] = 0.15
NEUER_WEIGHTS["avg_pass_distance"] = 0.15

NEUER_DIRECTIONS = GOALKEEPER_DIRECTIONS.copy()

# Goalkeeper archetypes - all computed from StatsBomb open data
# Source: https://github.com/statsbomb/open-data
# Script: scripts/compute_archetype_profiles.py
GOALKEEPER_ARCHETYPES = {
    "neuer": create_goalkeeper_archetype(
        "neuer",
        "Manuel Neuer (GER) - Modern Sweeper\n\nComputed from 2 Euro 2024 matches. "
        "90% pass accuracy, 46m average, 25% long balls. Excellent accuracy with balanced distribution.",
        {
            # Computed from StatsBomb open data (Euro 2024)
            "pass_success_rate": 90,      # Very high accuracy
            "long_pass_pct": 25,          # Moderate long balls
            "short_pass_pct": 45,         # Estimated (100 - long - medium)
            "avg_pass_distance": 46,      # Medium-long
            "quick_distribution_pct": 50, # Estimated
            "high_pass_pct": 30,          # Estimated
        },
        NEUER_WEIGHTS,
        NEUER_DIRECTIONS,
    ),
    "lloris": create_goalkeeper_archetype(
        "lloris",
        "Hugo Lloris (FRA) - Direct Distributor\n\nComputed from 2 World Cup 2022 matches. "
        "55% pass accuracy, 83m average, 69% long balls. France's direct style bypasses midfield.",
        {
            # Computed from StatsBomb open data (World Cup 2022)
            "pass_success_rate": 55,      # Lower accuracy (direct style)
            "long_pass_pct": 69,          # Mostly long balls
            "short_pass_pct": 20,         # Estimated
            "avg_pass_distance": 83,      # Very long distribution
            "quick_distribution_pct": 40, # Estimated
            "high_pass_pct": 60,          # Estimated (lofted balls)
        },
        LLORIS_WEIGHTS,
        LLORIS_DIRECTIONS,
    ),
    "bounou": create_goalkeeper_archetype(
        "bounou",
        "Yassine Bounou (MAR) - Balanced Long Distributor\n\nComputed from 5 World Cup 2022 matches. "
        "72% pass accuracy, 70m average, 50% long balls. Morocco's reliable distribution.",
        {
            # Computed from StatsBomb open data (World Cup 2022)
            "pass_success_rate": 72,      # Good accuracy
            "long_pass_pct": 50,          # Half long balls
            "short_pass_pct": 30,         # Estimated
            "avg_pass_distance": 70,      # Long distribution
            "quick_distribution_pct": 45, # Estimated
            "high_pass_pct": 45,          # Estimated
        },
        BOUNOU_WEIGHTS,
        BOUNOU_DIRECTIONS,
    ),
}


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


# Load data based on position
if position == "Forwards":
    profiles, n_events, n_matches = load_forward_data()
    event_type = "entries"
    position_type = "forward"

    st.sidebar.header("Forward Archetype")
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in FORWARD_ARCHETYPES],
        index=0,
    )
    selected_key = next(key for label, key in FORWARD_ARCHETYPES if label == selected_label)
    archetype = load_forward_archetype(selected_key)

    display_cols = [
        "rank", "player_name", "age", "team_name", "similarity_score",
        "total_entries", "danger_rate", "avg_separation"
    ]
    col_names = ["Rank", "Player", "Age", "Team", "Similarity %", "Entries", "Danger %", "Separation (m)"]
    radar_features = ["danger_rate", "central_pct", "avg_separation", "avg_entry_speed", "carry_pct"]
    caption = "Forward archetypes computed from StatsBomb World Cup 2022 event data."

elif position == "Defenders":
    profiles, n_events, n_matches = load_defender_data()
    event_type = "engagements"
    position_type = "defender"

    st.sidebar.header("Defender Archetype")
    defender_options = [
        ("Gvardiol (CRO) - Ball-playing CB", "gvardiol"),
        ("Van Dijk (NED) - Commanding CB", "vandijk"),
        ("Hakimi (MAR) - Attacking Wing-back", "hakimi"),
    ]
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in defender_options],
        index=0,
    )
    selected_key = next(key for label, key in defender_options if label == selected_label)
    # Archetypes computed from StatsBomb open data (World Cup 2022, Euro 2024)
    archetype = DEFENDER_ARCHETYPES[selected_key]

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
    gk_options = [
        ("Neuer (GER) - Modern Sweeper", "neuer"),
        ("Lloris (FRA) - Direct Distributor", "lloris"),
        ("Bounou (MAR) - Balanced Long", "bounou"),
    ]
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in gk_options],
        index=0,
    )
    selected_key = next(key for label, key in gk_options if label == selected_label)
    # Archetypes: Neuer computed from StatsBomb, others estimated
    archetype = GOALKEEPER_ARCHETYPES[selected_key]

    display_cols = [
        "rank", "player_name", "age", "team_name", "similarity_score",
        "total_distributions", "pass_success_rate", "long_pass_pct"
    ]
    col_names = ["Rank", "Player", "Age", "Team", "Similarity %", "Distributions", "Pass Success %", "Long Pass %"]
    radar_features = ["pass_success_rate", "avg_pass_distance", "long_pass_pct", "quick_distribution_pct"]
    caption = "All archetypes computed from StatsBomb open data (Euro 2024 / World Cup 2022)."

top_n = st.sidebar.slider("Number of results", min_value=5, max_value=min(20, len(profiles)), value=min(10, len(profiles)))

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Stats")
st.sidebar.markdown(f"**Events:** {n_events:,} {event_type}")
st.sidebar.markdown(f"**Matches:** {n_matches}")
st.sidebar.markdown(f"**Players:** {len(profiles)} qualified")

# Methodology note
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

# View selector (persists across reruns)
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
