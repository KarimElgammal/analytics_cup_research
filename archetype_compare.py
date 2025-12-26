"""Interactive Archetype Comparison Tool.

A Streamlit app for comparing A-League players against different player archetypes.
Supports forwards (final third entries), defenders (defensive engagements),
and goalkeepers (distribution).

Run: streamlit run archetype_compare.py
"""

import warnings
import streamlit as st
import polars as pl

from src.data.loader import load_all_events, add_team_names
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles
from src.analysis.defenders import detect_defensive_actions, build_defender_profiles, filter_defender_profiles
from src.analysis.goalkeepers import detect_gk_actions, build_goalkeeper_profiles, filter_goalkeeper_profiles
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

st.set_page_config(
    page_title="Archetype Comparison",
    page_icon="\u26BD",
    layout="wide",
)

st.title("\u26BD Archetype Comparison Tool")

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

# Defender weights and directions
DEFENDER_WEIGHTS = {
    "stop_danger_rate": 0.25,
    "reduce_danger_rate": 0.15,
    "force_backward_rate": 0.10,
    "beaten_by_possession_rate": 0.15,
    "beaten_by_movement_rate": 0.10,
    "pressing_rate": 0.10,
    "goal_side_rate": 0.10,
    "avg_engagement_distance": 0.05,
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

# Goalkeeper weights and directions
GOALKEEPER_WEIGHTS = {
    "pass_success_rate": 0.25,
    "long_pass_pct": 0.15,
    "short_pass_pct": 0.15,
    "avg_pass_distance": 0.15,
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


def create_defender_archetype(name: str, description: str, profile: dict) -> Archetype:
    """Create a defender archetype."""
    return Archetype(
        name=name,
        description=description,
        target_profile=profile,
        weights=DEFENDER_WEIGHTS,
        directions=DEFENDER_DIRECTIONS,
    )


def create_goalkeeper_archetype(name: str, description: str, profile: dict, weights: dict, directions: dict) -> Archetype:
    """Create a goalkeeper archetype."""
    return Archetype(
        name=name,
        description=description,
        target_profile=profile,
        weights=weights,
        directions=directions,
    )


# Defender archetypes
DEFENDER_ARCHETYPES = {
    "gvardiol": create_defender_archetype(
        "gvardiol",
        "Josko Gvardiol (CRO) - Ball-playing CB\n\nAggressive defender who steps out to engage. "
        "High pressing rate, comfortable defending in middle third. Progressive ball carrier.",
        {
            "stop_danger_rate": 70,
            "reduce_danger_rate": 80,
            "force_backward_rate": 60,
            "beaten_by_possession_rate": 20,
            "beaten_by_movement_rate": 15,
            "pressing_rate": 75,
            "goal_side_rate": 85,
            "avg_engagement_distance": 30,
            "middle_third_pct": 40,
        }
    ),
    "romero": create_defender_archetype(
        "romero",
        "Cristian Romero (ARG) - Aggressive CB\n\nFront-foot defender, aggressive in duels. "
        "Very high engagement rate, willing to follow attackers. Dominant in 1v1 situations.",
        {
            "stop_danger_rate": 80,
            "reduce_danger_rate": 70,
            "force_backward_rate": 75,
            "beaten_by_possession_rate": 25,
            "beaten_by_movement_rate": 20,
            "pressing_rate": 85,
            "goal_side_rate": 80,
            "avg_engagement_distance": 25,
            "defensive_third_pct": 50,
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

GOALKEEPER_ARCHETYPES = {
    "lloris": create_goalkeeper_archetype(
        "lloris",
        "Hugo Lloris (FRA) - Sweeper Keeper\n\nExcellent with feet, prefers playing out from the back. "
        "High pass success rate, quick distribution to maintain possession. "
        "Comfortable with short passes under pressure.",
        {
            "pass_success_rate": 85,
            "short_pass_pct": 70,
            "long_pass_pct": 20,
            "avg_pass_distance": 25,
            "quick_distribution_pct": 60,
            "to_middle_third_pct": 50,
            "avg_passing_options": 70,
        },
        LLORIS_WEIGHTS,
        LLORIS_DIRECTIONS,
    ),
    "livakovic": create_goalkeeper_archetype(
        "livakovic",
        "Dominik Livakovic (CRO) - Traditional Keeper\n\nStrong long distribution, finds forwards quickly. "
        "Comfortable launching attacks with long balls. "
        "Direct style bypasses midfield press.",
        {
            "pass_success_rate": 75,
            "long_pass_pct": 60,
            "short_pass_pct": 25,
            "avg_pass_distance": 45,
            "quick_distribution_pct": 40,
            "high_pass_pct": 50,
            "to_attacking_third_pct": 20,
        },
        LIVAKOVIC_WEIGHTS,
        LIVAKOVIC_DIRECTIONS,
    ),
    "bounou": create_goalkeeper_archetype(
        "bounou",
        "Yassine Bounou (MAR) - Long Distribution Specialist\n\nMorocco's World Cup hero with exceptional long distribution. "
        "39m average pass length, 42% high balls. Direct style finds forwards quickly. "
        "Famous penalty specialist (2 saves vs Spain).",
        {
            "pass_success_rate": 74,
            "long_pass_pct": 65,
            "short_pass_pct": 20,
            "avg_pass_distance": 50,
            "high_pass_pct": 55,
            "quick_distribution_pct": 35,
            "to_attacking_third_pct": 25,
        },
        BOUNOU_WEIGHTS,
        BOUNOU_DIRECTIONS,
    ),
}


@st.cache_data(ttl=3600, show_spinner="Loading A-League data...")
def load_forward_data():
    """Load and prepare forward profiles from final third entries."""
    events = load_all_events()
    events = add_team_names(events)
    entries = detect_entries(events)
    entries = classify_entries(entries)
    profiles = build_player_profiles(entries)
    profiles = filter_profiles(profiles, min_entries=3)
    return profiles, len(entries), events["match_id"].n_unique()


@st.cache_data(ttl=3600, show_spinner="Loading A-League defensive data...")
def load_defender_data():
    """Load and prepare defender profiles from defensive engagements."""
    events = load_all_events()
    events = add_team_names(events)
    actions = detect_defensive_actions(events)
    profiles = build_defender_profiles(actions)
    profiles = filter_defender_profiles(profiles, min_engagements=5)
    return profiles, len(actions), events["match_id"].n_unique()


@st.cache_data(ttl=3600, show_spinner="Loading A-League goalkeeper data...")
def load_goalkeeper_data():
    """Load and prepare goalkeeper profiles from distribution events."""
    events = load_all_events()
    events = add_team_names(events)
    actions = detect_gk_actions(events)
    profiles = build_goalkeeper_profiles(actions)
    profiles = filter_goalkeeper_profiles(profiles, min_distributions=10)
    return profiles, len(actions), events["match_id"].n_unique()


@st.cache_data(ttl=3600, show_spinner="Loading archetype...")
def load_forward_archetype(player_key: str):
    """Load forward archetype from StatsBomb data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Archetype.from_statsbomb(player_key, verbose=False)


# Load data based on position
if position == "Forwards":
    profiles, n_events, n_matches = load_forward_data()
    event_type = "entries"

    st.sidebar.header("Forward Archetype")
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in FORWARD_ARCHETYPES],
        index=0,
    )
    selected_key = next(key for label, key in FORWARD_ARCHETYPES if label == selected_label)
    archetype = load_forward_archetype(selected_key)

    display_cols = [
        "rank", "player_name", "team_name", "similarity_score",
        "total_entries", "danger_rate", "avg_separation"
    ]
    col_names = ["Rank", "Player", "Team", "Similarity %", "Entries", "Danger %", "Separation (m)"]
    caption = "Forward archetypes computed from StatsBomb World Cup 2022 event data."

elif position == "Defenders":
    profiles, n_events, n_matches = load_defender_data()
    event_type = "engagements"

    st.sidebar.header("Defender Archetype")
    defender_options = [
        ("Gvardiol (CRO) - Ball-playing CB", "gvardiol"),
        ("Romero (ARG) - Aggressive CB", "romero"),
    ]
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in defender_options],
        index=0,
    )
    selected_key = next(key for label, key in defender_options if label == selected_label)
    archetype = DEFENDER_ARCHETYPES[selected_key]

    display_cols = [
        "rank", "player_name", "team_name", "similarity_score",
        "total_engagements", "stop_danger_rate", "pressing_rate"
    ]
    col_names = ["Rank", "Player", "Team", "Similarity %", "Engagements", "Stop Danger %", "Pressing %"]
    caption = "Defender archetypes based on known playing styles. Profiles from on-ball engagement events."

else:  # Goalkeepers
    profiles, n_events, n_matches = load_goalkeeper_data()
    event_type = "distributions"

    st.sidebar.header("Goalkeeper Archetype")
    gk_options = [
        ("Lloris (FRA) - Sweeper Keeper", "lloris"),
        ("Livakovic (CRO) - Traditional Keeper", "livakovic"),
        ("Bounou (MAR) - Long Distribution", "bounou"),
    ]
    selected_label = st.sidebar.selectbox(
        "Select Archetype",
        options=[label for label, _ in gk_options],
        index=0,
    )
    selected_key = next(key for label, key in gk_options if label == selected_label)
    archetype = GOALKEEPER_ARCHETYPES[selected_key]

    display_cols = [
        "rank", "player_name", "team_name", "similarity_score",
        "total_distributions", "pass_success_rate", "long_pass_pct"
    ]
    col_names = ["Rank", "Player", "Team", "Similarity %", "Distributions", "Pass Success %", "Long Pass %"]
    caption = "Goalkeeper archetypes based on distribution style. Profiles from possession events."

top_n = st.sidebar.slider("Number of results", min_value=5, max_value=min(20, len(profiles)), value=min(10, len(profiles)))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data:** {n_events} {event_type} from {n_matches} matches")
st.sidebar.markdown(f"**Players:** {len(profiles)} with sufficient data")

# Run similarity engine
engine = SimilarityEngine(archetype)
engine.fit(profiles)
results = engine.rank(top_n=top_n)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Top {top_n} A-League {position}")
    st.markdown(f"*Matching: {archetype.name.title()} archetype*")

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
        width="stretch",
    )

with col2:
    st.subheader("Archetype Profile")
    st.markdown(archetype.description)

    st.markdown("**Feature Weights:**")
    importances = engine.get_feature_importance()
    for item in importances[:5]:
        pct = item["weight"] * 100
        st.markdown(f"- `{item['feature']}`: {pct:.0f}%")

st.markdown("---")
st.caption(caption + " Similarity uses weighted cosine similarity on z-score normalised features.")
