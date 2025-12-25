# Methodology: Finding Alvarez in the A-League

## Overview

This document explains how I derive the Julian Alvarez archetype from StatsBomb event data and map it to SkillCorner tracking metrics for player similarity analysis.

---

## Alvarez Target Profile for Radar Comparison

Since Alvarez does not appear in the SkillCorner A-League dataset, I constructed a target profile by mapping his StatsBomb event metrics to estimated SkillCorner tracking equivalents. Values are expressed on a normalised 0-100 scale.

| Feature | Target Value | Derivation |
|---------|--------------|------------|
| avg_separation | 85 | StatsBomb shows 24 box touches, indicating consistent dangerous positioning. High separation from defenders required. Target set at ~95th percentile of A-League data. |
| danger_rate | 90 | StatsBomb: 20% conversion, 60% shot accuracy in World Cup/Copa América. Elite finishing. Target at 95th percentile. |
| central_pct | 75 | Positional data shows Alvarez operates centrally, not as wide forward. Box touches come from central runs. |
| avg_entry_speed | 70 | Dynamic but not pace-reliant. Value comes from timing and positioning rather than raw speed. |
| half_space_pct | 60 | Complements central play. Drifts into half-spaces but primary threat through middle. |
| avg_passing_options | 65 | StatsBomb: 2 key passes, 78.9% pass accuracy. Good link-up player. |
| carry_pct | 40 | CRITICAL: 50% dribble success. Alvarez is NOT a dribbler. Low target penalises dribble-reliant players. |

The low carry_pct target is particularly important. While almost all A-League entries are classified as carries (98.8% mean), setting a low target means players who rely heavily on dribbling will score lower on this dimension. Alvarez creates through movement off the ball, not carrying through defenders.

---

## Part 1: Extracting Alvarez Profile from StatsBomb

### StatsBomb Free Data

StatsBomb provides free event data including World Cup and Copa América matches where Alvarez played.

```python
"""Extract Alvarez profile from StatsBomb free data.
Based on: player-focus/alvarez_marmoush_comparison.py
"""
import polars as pl
import requests

STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
ALVAREZ_PLAYER_ID = 29560  # Found in Argentina lineup

def find_player_events_by_id(player_id: int, player_name: str) -> pl.DataFrame:
    """Find all events for a specific player ID across all competitions."""
    competitions_url = f"{STATSBOMB_BASE_URL}competitions.json"
    competitions = requests.get(competitions_url).json()

    all_events = []
    for comp in competitions:
        comp_id, season_id = comp['competition_id'], comp['season_id']
        try:
            matches_url = f"{STATSBOMB_BASE_URL}matches/{comp_id}/{season_id}.json"
            matches = requests.get(matches_url).json()

            for match in matches:
                match_id = match['match_id']
                # Check lineup for player
                lineup_url = f"{STATSBOMB_BASE_URL}lineups/{match_id}.json"
                lineups = requests.get(lineup_url).json()

                player_in_match = any(
                    player.get('player_id') == player_id
                    for team in lineups for player in team.get('lineup', [])
                )

                if player_in_match:
                    events_url = f"{STATSBOMB_BASE_URL}events/{match_id}.json"
                    events = requests.get(events_url).json()
                    player_events = [e for e in events if e.get('player', {}).get('id') == player_id]
                    all_events.extend(player_events)
        except Exception:
            continue

    return pl.DataFrame(all_events) if all_events else pl.DataFrame()


def calculate_player_stats(events_df: pl.DataFrame) -> dict:
    """Calculate key performance metrics from StatsBomb events."""
    stats = {
        'total_events': len(events_df),
        'goals': 0, 'shots': 0, 'shots_on_target': 0,
        'passes_completed': 0, 'passes_attempted': 0,
        'dribbles_completed': 0, 'dribbles_attempted': 0,
        'touches_in_box': 0, 'key_passes': 0
    }

    for row in events_df.iter_rows(named=True):
        event_type = row.get('type', {})
        event_name = event_type.get('name', '') if isinstance(event_type, dict) else str(event_type)

        if event_name == 'Shot':
            stats['shots'] += 1
            outcome = row.get('shot', {}).get('outcome', {})
            if isinstance(outcome, dict):
                if outcome.get('name') == 'Goal':
                    stats['goals'] += 1
                if outcome.get('name') in ['Goal', 'Saved']:
                    stats['shots_on_target'] += 1

        elif event_name == 'Pass':
            stats['passes_attempted'] += 1
            if row.get('pass', {}).get('outcome') is None:
                stats['passes_completed'] += 1
            if row.get('pass', {}).get('shot_assist'):
                stats['key_passes'] += 1

        elif event_name == 'Dribble':
            stats['dribbles_attempted'] += 1
            outcome = row.get('dribble', {}).get('outcome', {})
            if isinstance(outcome, dict) and outcome.get('name') == 'Complete':
                stats['dribbles_completed'] += 1

        # Box touches
        if event_name in ['Shot', 'Pass', 'Ball Receipt*', 'Carry']:
            location = row.get('location', [])
            if len(location) >= 2 and location[0] >= 102:
                stats['touches_in_box'] += 1

    # Compute rates
    if stats['passes_attempted'] > 0:
        stats['pass_accuracy'] = stats['passes_completed'] / stats['passes_attempted'] * 100
    if stats['dribbles_attempted'] > 0:
        stats['dribble_success'] = stats['dribbles_completed'] / stats['dribbles_attempted'] * 100
    if stats['shots'] > 0:
        stats['shot_accuracy'] = stats['shots_on_target'] / stats['shots'] * 100
        stats['conversion_rate'] = stats['goals'] / stats['shots'] * 100

    return stats


# Run extraction
alvarez_events = find_player_events_by_id(ALVAREZ_PLAYER_ID, "Julián Álvarez")
alvarez_stats = calculate_player_stats(alvarez_events)
print(alvarez_stats)
```

### Alvarez Profile Results (from StatsBomb)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Events | 342 | Sample across WC/Copa |
| Shot Accuracy | 60.0% | Clinical finisher |
| Conversion Rate | 20.0% | High-pressure performer |
| Pass Accuracy | 78.9% | Takes calculated risks |
| Dribble Success | 50.0% | NOT a dribbler |
| Box Touches | 24 | Comfortable in danger |
| Key Passes | 2 | Creates as well as finishes |

---

## Part 2: Mapping StatsBomb → SkillCorner

### The Challenge

StatsBomb provides **event data** (shots, passes, dribbles), while SkillCorner provides **tracking data** (positions, speeds, separations). We must map concepts, not metrics directly.

### Mapping Table

| Alvarez Trait (StatsBomb) | SkillCorner Metric | Rationale |
|---------------------------|-------------------|-----------|
| 20% conversion rate | `danger_rate` | Entries leading to shots proxy finishing quality |
| 24 box touches | `central_pct` | Central entries = box presence |
| Intelligent movement | `avg_separation` | Finding space between lines |
| NOT a dribbler (50%) | `carry_pct` (LOW weight) | Carries ≠ dribbles in tracking |
| Link-up play (2 key passes) | `avg_passing_options` | More options = better link-up |
| High-pressure performer | `quick_break_pct` | Counter-attacks test composure |

---

## Part 2B: Loading A-League Data with Kloppy

### SkillCorner Open Data via Kloppy

Kloppy provides a standardised way to load SkillCorner tracking data:

```python
"""Load SkillCorner A-League data with kloppy."""
from kloppy import skillcorner

# Load tracking data for a match (frame-by-frame positions)
dataset = skillcorner.load_open_data(
    match_id=2017461,  # Melbourne Victory vs Auckland FC
    include_event_data=True
)

# Access frames
for frame in dataset:
    # frame.ball_coordinates
    # frame.players_data (dict of player_id -> PlayerData)
    pass

# Convert to DataFrame
df = dataset.to_df()
```

### Why We Use Dynamic Events Instead

For this research, I use the **dynamic_events.csv** files directly because they contain pre-computed game intelligence metrics:

```python
"""Load SkillCorner dynamic events (game intelligence)."""
import polars as pl

GITHUB_BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
MATCH_IDS = [2017461, 2015213, 2013725, 2011166, 2006229,
             1996435, 1953632, 1925299, 1899585, 1886347]

def load_events(match_id: int) -> pl.DataFrame:
    """Load dynamic events for a single match."""
    url = f"{GITHUB_BASE_URL}/matches/{match_id}/{match_id}_dynamic_events.csv"
    return pl.read_csv(url, infer_schema_length=10000)

def load_all_events() -> pl.DataFrame:
    """Load all matches."""
    dfs = [load_events(mid).with_columns(pl.lit(mid).alias("match_id"))
           for mid in MATCH_IDS]
    return pl.concat(dfs, how="diagonal")

# Available columns include:
# - speed_avg, distance_covered (physical)
# - separation_end, delta_to_last_defensive_line_end (spatial)
# - n_passing_options_ahead, n_teammates_ahead_end (tactical)
# - third_start, third_end, channel_end (zones)
# - lead_to_shot, lead_to_goal (outcomes)
```

### Detecting Final Third Entries

```python
def detect_entries(events: pl.DataFrame) -> pl.DataFrame:
    """Detect final third entries."""
    return events.filter(
        (pl.col("event_type") == "player_possession") &
        (pl.col("third_start") != "attacking_third") &
        (pl.col("third_end") == "attacking_third")
    )
```

---

## Part 3: Data-Driven Weight Calibration

### SkillCorner A-League Statistics

```python
"""Compute correlations to calibrate weights."""
import polars as pl
import numpy as np

# Load A-League data
from src.data.loader import load_all_events, add_team_names
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles

events = load_all_events()
events = add_team_names(events)
entries = detect_entries(events)
entries = classify_entries(entries)
profiles = build_player_profiles(entries)
profiles = filter_profiles(profiles, min_entries=3)

# Compute correlations with danger_rate
features = ['avg_separation', 'central_pct', 'half_space_pct',
            'avg_entry_speed', 'avg_passing_options', 'carry_pct']

danger = profiles['danger_rate'].to_numpy()
for feat in features:
    vals = profiles[feat].to_numpy()
    corr = np.corrcoef(danger, vals)[0, 1]
    print(f'{feat}: r={corr:+.3f}')
```

### Correlation Results (A-League 2024/25)

| Feature | Correlation with danger_rate | Weight Decision |
|---------|------------------------------|-----------------|
| `avg_entry_speed` | **r=+0.340** | **20%** - strongest predictor |
| `central_pct` | **r=+0.295** | **18%** - central is dangerous |
| `quick_break_pct` | r=+0.262 | 10% - counter-attack threat |
| `avg_separation` | r=+0.221 | 15% - intelligent movement |
| `avg_defensive_line_dist` | r=-0.259 | 12% - closer to goal |
| `half_space_pct` | r=-0.012 | **3%** - no correlation! |
| `avg_passing_options` | r=-0.050 | **2%** - no correlation! |
| `carry_pct` | mean=98.8% | **1%** - no variance! |

### Key Insight: carry_pct is Useless

```
carry_pct: mean=98.82%, std=4.60%
```

Almost ALL final third entries in A-League are carries. This feature has virtually no discriminating power and should be weighted near-zero.

---

## Part 4: Final Weight Configuration

### Weights Justified by Data + Archetype

```python
FEATURE_WEIGHTS = {
    # DATA-DRIVEN: Strongest correlations with danger_rate
    "avg_entry_speed": 0.20,          # r=+0.340 strongest predictor
    "central_pct": 0.18,              # r=+0.295 central = dangerous
    "avg_separation": 0.15,           # r=+0.221 intelligent movement
    "avg_defensive_line_dist": 0.12,  # r=-0.259 closer to goal

    # ARCHETYPE-DRIVEN: From StatsBomb profile
    "danger_rate": 0.15,              # 20% conversion (clinical)
    "quick_break_pct": 0.10,          # r=+0.262 counter-attack

    # LOW/ZERO CORRELATION - reduced
    "half_space_pct": 0.03,           # r=-0.012 useless
    "avg_passing_options": 0.02,      # r=-0.050 useless
    "avg_teammates_ahead": 0.03,      # r=-0.240 negative

    # NEAR-ZERO VARIANCE - effectively useless
    "carry_pct": 0.01,                # 98.8% mean, no variance
    "goal_rate": 0.00,                # Too sparse
}
```

---

## Part 5: Similarity Computation

### Weighted Cosine Similarity

```python
def compute_similarity(player_profile: dict, target_profile: dict, weights: dict) -> float:
    """
    Compute weighted cosine similarity between player and target.

    1. Z-score normalize all features
    2. Apply weights to both vectors
    3. Compute cosine similarity
    4. Scale to 0-100
    """
    features = list(weights.keys())

    # Normalize
    player_vec = normalize(player_profile, features)
    target_vec = normalize(target_profile, features)

    # Weight
    player_weighted = player_vec * weights
    target_weighted = target_vec * weights

    # Cosine similarity
    similarity = dot(player_weighted, target_weighted) / (norm(player_weighted) * norm(target_weighted))

    # Scale to 0-100
    return (similarity + 1) * 50
```

### Target Profile Definition

For features where higher is better (direction=1), target = 90th percentile.
For features where lower is better (direction=-1), target = 10th percentile.

---

---

## Part 6: ML Model for Weight Validation

### GradientBoosting Feature Importance

I trained a GradientBoosting classifier to predict which entries lead to shots:

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model = GradientBoostingClassifier(n_estimators=100, max_depth=4)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
# CV AUC: 0.656 ± 0.027
```

### ML Feature Importances

| Feature | ML Importance | Final Weight |
|---------|---------------|--------------|
| `separation_end` | **16.2%** | **23%** |
| `speed_avg` | 11.8% | 17% |
| `delta_to_last_defensive_line` | 10.3% | 15% |
| `entry_zone_num` | 8.5% | 12% |
| `n_passing_options_ahead` | **1.2%** | 2% |
| `entry_method_num` | **0.0%** | 0% |

### Key Insight: Why Key Passes Don't Predict Danger

StatsBomb **key passes** (event-level: pass leads to shot) ≠ SkillCorner **passing options** (state-level: options available at entry).

- **StatsBomb**: Measures EXECUTION - did this pass create a chance?
- **SkillCorner**: Measures POTENTIAL - how many teammates ahead?

Alvarez's key pass ability is about **decision-making and execution**, which tracking data can't directly capture. We compensate by weighting `danger_rate` higher (18%) - entries leading to shots is the closest proxy.

---

## Summary

| Step | Data Source | Method |
|------|-------------|--------|
| 1. Define archetype | StatsBomb (WC/Copa) | Extract Alvarez event metrics |
| 2. Map to tracking | Conceptual | StatsBomb events → SkillCorner tracking |
| 3. Calibrate weights | ML (GradientBoosting) | Feature importance on A-League data |
| 4. Compute similarity | Combined | Weighted cosine similarity |

This hybrid approach combines:
- **Domain knowledge** (Alvarez archetype from StatsBomb)
- **ML validation** (GradientBoosting feature importance)
- **Correlation analysis** (A-League data patterns)
