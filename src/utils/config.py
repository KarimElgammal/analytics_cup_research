"""Configuration constants for the Analytics Cup Research Track submission."""

# GitHub data URLs
GITHUB_BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
GITHUB_LFS_URL = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"

# Match IDs with tracking data
MATCH_IDS = [
    2017461,  # Melbourne Victory vs Auckland FC
    2015213,  # Western United vs Auckland FC
    2013725,  # Western United vs Sydney FC
    2011166,  # Wellington Phoenix vs Melbourne Victory
    2006229,  # Melbourne City vs Macarthur FC
    1996435,  # Sydney FC vs Adelaide United
    1953632,  # CC Mariners vs Melbourne City
    1925299,  # Brisbane FC vs Perth Glory
    1899585,  # Auckland FC vs Wellington Phoenix
    1886347,  # Auckland FC vs Newcastle
]

# Team ID to name mapping
TEAMS = {
    4177: "Auckland FC",
    1805: "Newcastle Jets",
    867: "Wellington Phoenix",
    868: "Melbourne Victory",
    869: "Sydney FC",
    866: "Adelaide United",
    870: "Central Coast Mariners",
    2380: "Melbourne City",
    1804: "Macarthur FC",
    1803: "Western United",
    1802: "Brisbane Roar",
    871: "Perth Glory",
}

# Pitch dimensions (SkillCorner coordinate system)
PITCH_LENGTH = 105
PITCH_WIDTH = 68

# Entry zones
ENTRY_ZONES = {
    "central": ["center"],
    "half_space": ["half_space_left", "half_space_right"],
    "wide": ["wide_left", "wide_right"],
}

# Phase types
IN_POSSESSION_PHASES = [
    "build_up",
    "create",
    "finish",
    "direct",
    "quick_break",
    "transition",
    "set_play",
    "chaotic",
]
