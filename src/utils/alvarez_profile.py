"""Julian Alvarez archetype profile for player similarity matching.

This module defines the target profile for identifying A-League players
with Alvarez-like characteristics using SkillCorner tracking data.

The archetype is derived from StatsBomb event data analysis showing:
- 60% shot accuracy, 20% conversion rate (clinical finishing)
- 24 box touches per sample (comfortable in dangerous areas)
- 78.9% pass accuracy (takes calculated risks)
- 2 key passes (creates as well as finishes)
- Intelligent movement > raw dribbling (50% dribble success)

These event-based traits are mapped to SkillCorner tracking metrics below.
"""

# Features to compute from entry-level tracking data
PROFILE_FEATURES = [
    # Physical attributes
    "avg_entry_speed",       # Mean speed during entries (m/s)
    "avg_distance",          # Mean distance covered per entry (m)
    "total_entries",         # Volume of final third entries

    # Spatial positioning
    "avg_separation",        # Mean separation from defenders at entry end
    "avg_defensive_line_dist",  # Mean distance to last defensive line

    # Zone preferences (%)
    "central_pct",           # % of entries through central zone
    "half_space_pct",        # % of entries through half-spaces

    # Entry method (%)
    "carry_pct",             # % of entries via dribble/carry

    # Tactical context
    "avg_passing_options",   # Mean passing options ahead at entry end
    "avg_teammates_ahead",   # Mean teammates ahead at entry end

    # Outcomes (%)
    "danger_rate",           # % of entries leading to shots
    "goal_rate",             # % of entries leading to goals

    # Phase preferences (%)
    "quick_break_pct",       # % of entries during quick break phase
]

# Feature weights for similarity scoring
# Calibrated using ML (GradientBoosting) feature importance on A-League data
# Combined with StatsBomb Alvarez archetype traits
# CV AUC: 0.656 on 245 entries
FEATURE_WEIGHTS = {
    # ML-DERIVED: GradientBoosting feature importances (highest predictors of danger)
    "avg_separation": 0.23,           # ML: 16.2% importance - HIGHEST! Finds space
    "avg_entry_speed": 0.17,          # ML: 11.8% importance - dynamic entries
    "avg_defensive_line_dist": 0.15,  # ML: 10.3% importance - penetration depth
    "central_pct": 0.12,              # ML: 8.5% importance - central = dangerous

    # ARCHETYPE-DRIVEN: From StatsBomb Alvarez profile (20% conversion)
    "danger_rate": 0.18,              # Clinical finishing - core Alvarez trait

    # MODERATE IMPORTANCE
    "quick_break_pct": 0.05,          # ML: 3.7% importance - counter-attacks
    "avg_teammates_ahead": 0.05,      # ML: 3.8% importance - link-up context

    # ML CONFIRMED LOW/ZERO IMPORTANCE
    "half_space_pct": 0.02,           # Correlation near zero
    "avg_passing_options": 0.02,      # ML: 1.2% - key passes don't predict danger in tracking!
    "carry_pct": 0.00,                # ML: 0.0%! Confirms useless (no variance)
    "avg_distance": 0.01,             # Workrate
    "goal_rate": 0.00,                # Too sparse
}

# Target profile direction: 1 = higher is better, -1 = lower is better
FEATURE_DIRECTION = {
    "avg_entry_speed": 1,
    "avg_distance": 1,
    "total_entries": 0,               # Not directional (used for filtering)
    "avg_separation": 1,              # More separation = better
    "avg_defensive_line_dist": -1,    # Closer to goal = better (negative values)
    "central_pct": 1,
    "half_space_pct": 1,
    "carry_pct": 1,
    "avg_passing_options": 1,
    "avg_teammates_ahead": 1,
    "danger_rate": 1,
    "goal_rate": 1,
    "quick_break_pct": 1,
}

# Minimum entries required for reliable profile
MIN_ENTRIES_THRESHOLD = 3

# Alvarez archetype description for AI insights
ALVAREZ_DESCRIPTION = """
Julian Alvarez is a modern forward archetype characterised by intelligent movement,
spatial awareness, and clinical finishing rather than dribbling ability.

StatsBomb metrics (World Cup, Copa América):
- 60% shot accuracy, 20% conversion rate (clinical finisher)
- 24 box touches per tournament (comfortable in dangerous areas)
- 78.9% pass accuracy (calculated risk-taker in build-up)
- Only 50% dribble success (NOT a dribbler)
- 2 key passes (contributes to team play)

SkillCorner target profile mapping:
- danger_rate: 90/100 (mirrors 20% conversion)
- avg_separation: 85/100 (finds space between lines)
- central_pct: 75/100 (operates centrally, not wide)
- avg_entry_speed: 70/100 (dynamic but not pace-reliant)
- carry_pct: 40/100 (LOW - penalises dribblers)

The key insight: Alvarez creates danger through intelligent positioning and timing
rather than beating defenders with the ball. Similar players show high danger rates
and separation values without relying on dribbling.
"""


# Alvarez target profile - estimated SkillCorner values
# Derived by mapping StatsBomb event metrics to tracking equivalents
# Values expressed as normalised 0-100 scale for radar display
ALVAREZ_TARGET_PROFILE = {
    # HIGH VALUES - core Alvarez traits
    "avg_separation": 85,      # Intelligent movement, finds space (StatsBomb: 24 box touches)
    "danger_rate": 90,         # Clinical finishing (StatsBomb: 20% conversion, 60% shot accuracy)
    "central_pct": 75,         # Central preference (StatsBomb: box presence)

    # MODERATE-HIGH VALUES
    "avg_entry_speed": 70,     # Dynamic but not pace-reliant
    "half_space_pct": 60,      # Works half-spaces as well as central
    "avg_passing_options": 65, # Good link-up (StatsBomb: 2 key passes, 78.9% pass accuracy)

    # LOW VALUE - explicitly NOT an Alvarez trait
    "carry_pct": 40,           # NOT a dribbler (StatsBomb: 50% dribble success)
}

# Derivation methodology for Alvarez target values:
#
# 1. avg_separation (85/100):
#    StatsBomb shows Alvarez with 24 box touches per tournament sample, indicating
#    he consistently finds dangerous positions. This requires high separation from
#    defenders. Target set at ~95th percentile of A-League data.
#
# 2. danger_rate (90/100):
#    StatsBomb: 20% conversion rate, 60% shot accuracy in World Cup/Copa America.
#    Elite finishing ability. Target set at 95th percentile since A-League forwards
#    average ~15% danger rate.
#
# 3. central_pct (75/100):
#    StatsBomb positional data shows Alvarez operates centrally, not as a wide forward.
#    His box touches come from central runs, not crossing positions.
#
# 4. avg_entry_speed (70/100):
#    Alvarez is dynamic but not a pace merchant. His value comes from timing and
#    positioning rather than raw speed. Moderate-high target.
#
# 5. half_space_pct (60/100):
#    Complements central play. Alvarez drifts into half-spaces but primary threat
#    is through the middle.
#
# 6. avg_passing_options (65/100):
#    StatsBomb: 2 key passes, 78.9% pass accuracy. Good link-up player who creates
#    for others as well as finishing.
#
# 7. carry_pct (40/100):
#    CRITICAL: StatsBomb shows only 50% dribble success. Alvarez is NOT a dribbler.
#    He creates through movement off the ball, not carrying through defenders.
#    Low target value penalises players who rely on dribbling.


def get_alvarez_target() -> dict:
    """Return Alvarez target profile for radar comparison."""
    return ALVAREZ_TARGET_PROFILE.copy()


def get_weighted_features() -> list[str]:
    """Return list of features used in similarity scoring (excludes total_entries)."""
    return [f for f in PROFILE_FEATURES if f in FEATURE_WEIGHTS]


def get_feature_weight(feature: str) -> float:
    """Get weight for a feature, default 0 if not defined."""
    return FEATURE_WEIGHTS.get(feature, 0.0)


def get_feature_direction(feature: str) -> int:
    """Get direction for a feature (1=higher better, -1=lower better, 0=neutral)."""
    return FEATURE_DIRECTION.get(feature, 1)
