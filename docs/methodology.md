# Methodology: Finding Alvarez in the A-League

> **Online Documentation**: [karimelgammal.github.io/analytics_cup_research](https://karimelgammal.github.io/analytics_cup_research/)
>
> **Repository**: [github.com/KarimElgammal/analytics_cup_research](https://github.com/KarimElgammal/analytics_cup_research)

## Overview

This document explains how I derive player archetypes from StatsBomb event data and map them to SkillCorner tracking metrics for player similarity analysis.

---

## Data Pipeline Transparency

### What's Computed vs Static

| Component | Source | Type | Notes |
|-----------|--------|------|-------|
| **Forward archetypes** | StatsBomb API | Computed at runtime | Stats from World Cup 2022 events |
| **Defender archetypes** | StatsBomb open data | Computed locally | Gvardiol, Van Dijk, Hakimi from WC22/Euro24 |
| **Goalkeeper archetypes** | StatsBomb open data | Computed locally | Neuer, Lloris, Bounou from WC22/Euro24 |
| **A-League player profiles** | SkillCorner Dynamic Events | Computed at runtime | From `dynamic_events.csv` (A-League 2024/25) |
| **Feature weights** | Correlation analysis | Static | Derived from A-League data, hardcoded |
| **Reference distributions** | Manual estimates | Static | Percentile thresholds for mapping |

### Archetype Computation Details

| Position | Players | Source | Data |
|----------|---------|--------|------|
| **Forwards** | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri | StatsBomb API | World Cup 2022 events (shots, goals, dribbles) |
| **Defenders** | Gvardiol, Van Dijk, Hakimi | [StatsBomb open data](https://github.com/statsbomb/open-data) | WC22 + Euro24 events (duels, pressures, blocks) |
| **Goalkeepers** | Neuer, Lloris, Bounou | [StatsBomb open data](https://github.com/statsbomb/open-data) | WC22 + Euro24 passes |

**Script**: `scripts/compute_archetype_profiles.py` computes defender/goalkeeper metrics from StatsBomb open data.

### Sample Sizes

| Player | Competition | Matches | Events |
|--------|-------------|---------|--------|
| Gvardiol | WC22 | 3 | 48 defensive |
| Van Dijk | Euro24 | 3 | 42 defensive |
| Hakimi | WC22 | 6 | 148 defensive |
| Neuer | Euro24 | 2 | 59 passes |
| Lloris | WC22 | 2 | 69 passes |
| Bounou | WC22 | 5 | 83 passes |

### Mapping Event Data → Tracking Data

**Event Data** (StatsBomb - World Cup 2022, Euro 2024):
- Discrete on-ball actions: duels, pressures, passes, shots
- What happened and when

**Tracking Data** (SkillCorner - A-League 2024/25):
- Positional data from optical tracking at 25Hz
- Dynamic Events derived from tracking: entries, engagements, distributions

The mapping assumes behavioural consistency across data types:

| Event Data (StatsBomb) | → | Tracking Data (SkillCorner) |
|------------------------|---|------------------------------|
| High duel success rate | → | High `stop_danger_rate` |
| High pressure count | → | High `pressing_rate` |
| High clearance/block rate | → | High `goal_side_rate` |
| Long pass distance | → | High `avg_pass_distance` |

This is an approximation. A player with 67% duel success in event data should match players in tracking data who also stop ~67% of dangerous situations.

### Similarity Method

The tool uses **weighted cosine similarity** on z-score normalised features:

1. Compute correlations between features and outcomes (danger_rate, etc.)
2. Set weights based on correlation strength
3. Compare player profiles to archetype targets using cosine similarity

### Forward Target Features

| Feature | Source | Type |
|---------|--------|------|
| `danger_rate` | Conversion rate from StatsBomb | Computed |
| `avg_separation` | Box touches per 90 from StatsBomb | Computed |
| `carry_pct` | Dribble success from StatsBomb | Computed |
| `avg_passing_options` | Pass accuracy from StatsBomb | Computed |
| `central_pct` | No StatsBomb equivalent | Fixed estimate (70) |
| `avg_entry_speed` | No StatsBomb equivalent | Fixed estimate (65) |
| `half_space_pct` | No StatsBomb equivalent | Fixed estimate (55) |
| `avg_defensive_line_dist` | No StatsBomb equivalent | Fixed estimate (50) |

### Defender Target Features

Computed using `scripts/compute_archetype_profiles.py` from StatsBomb open data events.

| Feature | Calculation | Source |
|---------|-------------|--------|
| `stop_danger_rate` | `won_duels / total_duels * 100` where duel_outcome in ["Won", "Success In Play", "Success Out"] | StatsBomb |
| `reduce_danger_rate` | `(interceptions + ball_recoveries) / (total_defensive + ball_recoveries) * 100` | StatsBomb |
| `pressing_rate` | `pressures / total_defensive_events * 100` | StatsBomb |
| `goal_side_rate` | `(clearances + blocks) / total_defensive * 100 * 2.5` (scaled) | StatsBomb |
| `beaten_by_movement_rate` | `100 - stop_danger_rate` | StatsBomb |
| `avg_engagement_distance` | `avg(location[0]) / 120 * 100` for Pressure/Duel/Interception events | StatsBomb |
| `force_backward_rate` | Based on tackle patterns | Estimated |
| `beaten_by_possession_rate` | Based on duel loss patterns | Estimated |
| `middle_third_pct` | Zone distribution | Estimated |

### Goalkeeper Target Features

Computed from StatsBomb Pass events for goalkeeper players.

| Feature | Calculation | Source |
|---------|-------------|--------|
| `pass_success_rate` | `successful_passes / total_passes * 100` (null outcome = success) | StatsBomb |
| `avg_pass_distance` | `avg(pass_length) / 60 * 100` (normalised to 0-100) | StatsBomb |
| `long_pass_pct` | `passes_over_35m / total_passes * 100` | StatsBomb |
| `short_pass_pct` | Inverse of long pass tendency | Estimated |
| `high_pass_pct` | Lofted/aerial passes | Estimated |
| `quick_distribution_pct` | Speed of restarts | Estimated |

---

## Programmatic Archetype Generation

Archetypes are computed programmatically from StatsBomb event data.

```python
from src.core.archetype import Archetype

# Load archetype from StatsBomb data
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual stats

# Available players (12 archetypes across 3 positions)
Archetype.list_available()
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'vandijk', 'hakimi', 'neuer', 'lloris', 'bounou']
```

The `src/statsbomb/` package handles:

- **loader.py**: Fetches events from StatsBomb free data API
- **registry.py**: Maps player keys to StatsBomb names and competition IDs
- **stats.py**: Calculates shooting, passing, dribbling statistics
- **mapper.py**: Converts StatsBomb metrics to SkillCorner target profiles

---

## Alvarez Target Profile (Computed from StatsBomb)

The archetype is **computed from StatsBomb data**. Example output from World Cup 2022:

| Metric | Actual Value | Interpretation |
|--------|--------------|----------------|
| Shots | 11 | Active in attack |
| Goals | 4 | Clinical finisher |
| Conversion Rate | 36.4% | Elite finishing |
| Shot Accuracy | 72.7% | Very accurate |
| Dribble Success | 0.0% | NOT a dribbler |
| Box Touches | 8 | Finds dangerous positions |

The low dribble success is critical: Alvarez creates through **movement and positioning**, not individual dribbling. This maps to a low `carry_pct` target, penalising dribble-reliant players.

### Mapping to SkillCorner Targets

| StatsBomb Metric | SkillCorner Target | Percentile Mapping |
|------------------|--------------------|--------------------|
| Conversion Rate | `danger_rate` | 36.4% → ~95th percentile |
| Box Touches/90 | `avg_separation` | High box presence → good separation |
| Pass Accuracy | `avg_passing_options` | 78.9% → ~50th percentile |
| Dribble Success | `carry_pct` | 0% → very low target |

Fixed values for tracking-specific metrics (no StatsBomb equivalent):
- `central_pct`: 70 (forwards typically central)
- `avg_entry_speed`: 65 (moderate, not pace-reliant)
- `half_space_pct`: 55 (some half-space movement)

---

## Part 1: StatsBomb Integration

### Using the StatsBomb Package

The `src/statsbomb/` package provides a clean API for loading player data:

```python
from src.statsbomb.loader import StatsBombLoader
from src.statsbomb.stats import calculate_player_stats
from src.statsbomb.registry import get_player_info

# Initialize loader
loader = StatsBombLoader()

# Get player info
info = get_player_info("alvarez")
# {'player_name': 'Julián Álvarez', 'display_name': 'Julian Alvarez', ...}

# Load events from World Cup 2022
events = loader.get_player_events(
    player_name=info["player_name"],
    competitions=[(43, 106)]  # World Cup 2022
)

# Calculate statistics
stats = calculate_player_stats(events)
print(f"Conversion: {stats.conversion_rate:.1f}%")
print(f"Dribble Success: {stats.dribble_success:.1f}%")
```

### Player Registry

Available players are defined in `src/statsbomb/registry.py`:

```python
PLAYER_REGISTRY = {
    "alvarez": {
        "player_name": "Julián Álvarez",
        "display_name": "Julian Alvarez",
        "position": "Forward",
        "style": "Movement-focused, intelligent runs, clinical finishing",
        "competitions": [(43, 106, "FIFA World Cup 2022")],
    },
    "giroud": {
        "player_name": "Olivier Giroud",
        "display_name": "Olivier Giroud",
        "position": "Forward",
        "style": "Target man, hold-up play, aerial threat, 0% dribble reliance",
        "competitions": [(43, 106, "FIFA World Cup 2022")],
    },
    "kane": {
        "player_name": "Harry Kane",
        "display_name": "Harry Kane",
        "position": "Forward",
        "style": "Complete forward, link-up play, clinical finishing",
        "competitions": [(43, 106, "FIFA World Cup 2022")],
    },
    # ... plus lewandowski, rashford, en_nesyri, gvardiol, romero, hakimi, lloris, livakovic, bounou
}
```

### Computed Player Profiles (World Cup 2022)

| Player | Conversion | Shot Accuracy | Dribble Success | Style |
|--------|------------|---------------|-----------------|-------|
| Alvarez | 36.4% | 72.7% | 0.0% | Movement-focused |
| Giroud | 23.5% | 58.8% | 0.0% | Target man |
| Kane | 16.7% | 50.0% | 43% | Complete forward |
| Rashford | 27.3% | 54.5% | 60% | Pace + dribbling |

The key insight: Both Alvarez and Giroud have 0% dribble success. They create through movement, not dribbling. This distinguishes them from pace-reliant players like Rashford.

---

## Part 2: Mapping StatsBomb → SkillCorner

### The Challenge

StatsBomb provides **event data** (shots, passes, dribbles), while SkillCorner provides **tracking data** (positions, speeds, separations). The approach maps concepts, not metrics directly.

### Mapping Table

| Alvarez Trait (StatsBomb) | SkillCorner Metric | Rationale |
|---------------------------|-------------------|-----------|
| 36.4% conversion rate | `danger_rate` | Entries leading to shots proxy finishing quality |
| 8 box touches | `central_pct` | Central entries = box presence |
| Intelligent movement | `avg_separation` | Finding space between lines |
| NOT a dribbler (0%) | `carry_pct` (LOW weight) | Carries ≠ dribbles in tracking |
| Link-up play | `avg_passing_options` | More options = better link-up |
| High-pressure performer | `quick_break_pct` | Counter-attacks test composure |

---

## Part 2B: Loading A-League Data

### Loading Dynamic Events via Polars

I load the **dynamic_events.csv** files directly from SkillCorner's GitHub repository using Polars. These files contain pre-computed game intelligence metrics:

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

## Part 6: Weight Selection

Weights were chosen based on:

| Method | Used For |
|--------|----------|
| Correlation with `danger_rate` | Forward feature weights |
| Correlation with `stop_possession` | Defender feature weights |
| Correlation with `pass_success` | Goalkeeper feature weights |
| Domain knowledge | Feature directions (+1/-1) |

### Key Insight: Why Key Passes Don't Predict Danger

StatsBomb **key passes** (event-level: pass leads to shot) ≠ SkillCorner **passing options** (state-level: options available at entry).

- **StatsBomb**: Measures EXECUTION - did this pass create a chance?
- **SkillCorner**: Measures POTENTIAL - how many teammates ahead?

Alvarez's key pass ability is about **decision-making and execution**, which tracking data can't directly capture. The `danger_rate` feature (entries leading to shots) is the closest proxy.

---

## Summary

| Step | Data Source | Method |
|------|-------------|--------|
| 1. Define archetype | StatsBomb (WC 2022) | Extract player event metrics |
| 2. Map to tracking | Conceptual | StatsBomb events → SkillCorner features |
| 3. Choose weights | Correlation analysis | Feature importance from A-League data |
| 4. Compute similarity | Combined | Weighted cosine similarity |

This approach combines:

- **Domain knowledge** (player archetypes from StatsBomb World Cup 2022)
- **Correlation analysis** (A-League data patterns inform weight choices)
- **Simple similarity** (weighted cosine similarity, appropriate for small dataset)

---

## Part 7: Player Archetype Selection Criteria

### Why These 10 Players?

The archetypes were selected based on three key criteria:

1. **Data Availability**: Players must have sufficient events in StatsBomb free data (World Cup 2022) for reliable statistics
2. **Realistic Comparisons**: Avoiding exceptional generational talents (Messi, Mbappe) and focusing on high-calibre players whose profiles are more attainable for league-level scouting
3. **Style Diversity**: Covering different playing styles to demonstrate the tool's versatility

### Selection Process

I analysed World Cup 2022 data to find players with:

- **Forwards**: 5+ shots (sufficient sample for conversion rates)
- **Defenders**: 200+ passes, 10+ clearances/interceptions
- **Goalkeepers**: 10+ saves across multiple matches

### Available Archetypes

#### Forwards (6 players)

| Player | Country | Shots | Goals | Conv% | Dribbles | Style |
|--------|---------|-------|-------|-------|----------|-------|
| **Alvarez** | ARG | 11 | 4 | 36.4% | 0% | Movement-focused, intelligent runs |
| **Giroud** | FRA | 17 | 4 | 23.5% | 0% | Target man, hold-up play, aerial threat |
| **Kane** | ENG | 12 | 2 | 16.7% | 43% | Complete forward, link-up play |
| **Lewandowski** | POL | 12 | 2 | 16.7% | 40% | Clinical poacher, box presence |
| **Rashford** | ENG | 11 | 3 | 27.3% | 60% | Pace, direct dribbling, wide threat |
| **En-Nesyri** | MAR | 11 | 2 | 18.2% | 67% | Physical forward, aerial presence |

#### Defenders (3 players)

Computed from StatsBomb World Cup 2022 events:

| Player | Country | Matches | Tackles | Duels | Pressures | Style |
|--------|---------|---------|---------|-------|-----------|-------|
| **Gvardiol** | CRO | 7 | 15 (67%) | 23 (44%) | 77 (57/90) | Ball-playing CB |
| **Romero** | ARG | 7 | 12 (58%) | 28 (50%) | 65 (48/90) | Aggressive CB |
| **Hakimi** | MAR | 7 | 8 (50%) | 19 (42%) | 52 (39/90) | Attacking wing-back |

#### Goalkeepers (3 players)

Computed from StatsBomb World Cup 2022 events:

| Player | Country | Matches | Passes | Pass Success | Avg Distance | Long % |
|--------|---------|---------|--------|--------------|--------------|--------|
| **Lloris** | FRA | 6 | 185 | 78% | 28m | 35% |
| **Livakovic** | CRO | 7 | 195 | 72% | 35m | 45% |
| **Bounou** | MAR | 6 | 210 | 74% | 39m | 50% |

### Why Not Messi/Mbappe?

While Messi and Mbappe have extensive data in World Cup 2022, they represent **outlier profiles**:
- **Messi**: 72% dribble success, 26% conversion - unrealistic target for most leagues
- **Mbappe**: 60% dribble success, 28% conversion - pace/skill combination rare at any level

Using these as archetypes would yield few meaningful matches in A-League or similar leagues. The selected players represent **achievable excellence** - world-class but not once-in-a-generation.

---

## Part 8: Using This Tool with Other SkillCorner Data

### Architecture Overview

The tool is designed to work with **any SkillCorner tracking data**, not just A-League:

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR DATA                                                       │
│  - SkillCorner dynamic_events.csv                               │
│  - Any league/competition supported                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/data/loader.py                                              │
│  - load_events(match_id) or load_all_events()                   │
│  - Returns Polars DataFrame with tracking metrics                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/analysis/                                                   │
│  - entries.py: detect_entries(), classify_entries()            │
│  - profiles.py: build_player_profiles()                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/core/                                                       │
│  - archetype.py: Archetype.from_statsbomb("alvarez")            │
│  - similarity.py: SimilarityEngine(archetype).fit(profiles)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESULTS: Ranked players by similarity to archetype             │
└─────────────────────────────────────────────────────────────────┘
```

### Loading Your Own SkillCorner Data

#### Option 1: Use SkillCorner Open Data (other matches)

```python
from src.data.loader import load_events

# Load any match from SkillCorner open data
# See: github.com/SkillCorner/opendata for available match IDs
events = load_events(match_id=2017461)
```

#### Option 2: Use Local CSV Files

```python
import polars as pl

# Load your own SkillCorner dynamic_events.csv
events = pl.read_csv("path/to/your_dynamic_events.csv")

# Ensure required columns exist:
required = ["event_type", "player_id", "third_start", "third_end",
            "speed_avg", "separation_end", "lead_to_shot"]
```

### Running Similarity Analysis

```python
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# 1. Process your events
entries = detect_entries(your_events)
entries = classify_entries(entries)

# 2. Build player profiles
profiles = build_player_profiles(entries)
profiles = filter_profiles(profiles, min_entries=3)

# 3. Load any archetype
archetype = Archetype.from_statsbomb("kane")  # or giroud, lewandowski, etc.

# 4. Run similarity analysis
engine = SimilarityEngine(archetype)
engine.fit(profiles)
results = engine.rank(top_n=10)

print(results)
```

### Adding Custom Archetypes

You can create custom archetypes without StatsBomb data:

```python
from src.core.archetype import Archetype

# Create a custom archetype
pressing_forward = Archetype.custom(
    name="pressing_forward",
    description="High-energy forward who presses from the front"
)

# Set target profile (0-100 scale)
pressing_forward.set_feature("avg_entry_speed", target=85, weight=0.25, direction=1)
pressing_forward.set_feature("danger_rate", target=60, weight=0.20, direction=1)
pressing_forward.set_feature("avg_separation", target=70, weight=0.20, direction=1)
pressing_forward.set_feature("central_pct", target=80, weight=0.15, direction=1)
pressing_forward.set_feature("carry_pct", target=30, weight=0.10, direction=1)
pressing_forward.set_feature("quick_break_pct", target=70, weight=0.10, direction=1)

# Use with SimilarityEngine
engine = SimilarityEngine(pressing_forward)
engine.fit(profiles)
```

### Extending to New Leagues

To use with a new league's SkillCorner data:

1. **Obtain SkillCorner data** for your league (via SkillCorner API or open data)
2. **Load events** using `load_events()` or custom loader
3. **Run the pipeline** - entries → profiles → similarity
4. **Interpret results** considering league context

The feature weights (separation: 23%, danger_rate: 18%, etc.) were derived from A-League correlation analysis but should generalise reasonably to other leagues with similar playing styles.

---

## Part 9: Adding New Players to the Registry

To add a new player archetype from StatsBomb data:

### Step 1: Find the Player in StatsBomb

```python
from statsbombpy import sb

# Check available competitions
comps = sb.competitions()
print(comps[['competition_id', 'season_name', 'competition_name']])

# World Cup 2022 = competition_id: 43, season_id: 106
matches = sb.matches(competition_id=43, season_id=106)

# Get events and find player name
events = sb.events(match_id=matches['match_id'].iloc[0])
print(events['player'].unique())
```

### Step 2: Add to Registry

Edit `src/statsbomb/registry.py`:

```python
PLAYER_REGISTRY = {
    # ... existing players ...

    "your_player": {
        "player_name": "Exact Name From StatsBomb",  # Must match exactly!
        "display_name": "Display Name",
        "position": "Forward",  # or "Defender", "Goalkeeper"
        "nationality": "Country",
        "style": "Brief style description",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),  # (competition_id, season_id, name)
        ],
    },
}
```

### Step 3: Test

```python
from src.core.archetype import Archetype

arch = Archetype.from_statsbomb("your_player")
print(arch.description)  # Should show computed stats
```

---

## Part 10: AI-Powered Scouting Insights

### Overview

The tool includes optional AI-generated scouting recommendations using GitHub Models (Grok 3 Mini, Phi-4, GPT-4o Mini) or HuggingFace Inference API (Llama 3.1/3.2, Qwen 2.5, SmolLM3, Gemma 2). These insights contextualise the similarity rankings with human-readable analysis.

**Note:** Grok 3 Mini is a reasoning model that uses more tokens (and thus more cost) than standard models. Running locally has no rate limits.

### How It Works

The AI receives a position-aware prompt containing:

1. **Position context**: forward, defender, or goalkeeper
2. **Archetype description**: from StatsBomb profile
3. **Top candidates**: player names, ages, teams, and key metrics
4. **Dataset averages**: for comparison context
5. **Position criteria**: what makes a great player in this position

### Position-Specific Metrics

Different metrics matter for different positions:

| Position | Key Metrics | Events |
|----------|-------------|--------|
| Forward | danger_rate, central_pct, separation, entry_speed | 245 entries |
| Defender | stop_danger_rate, pressing_rate, goal_side_rate | 8,911 engagements |
| Goalkeeper | pass_success_rate, pass_distance, long_pass_pct | 522 distributions |

### Example Usage

```python
from src.utils.ai_insights import generate_similarity_insight, has_valid_token

if has_valid_token():
    insight = generate_similarity_insight(
        ranked_players,
        archetype,
        top_n=5,
        position_type="forward",
    )
    print(insight)
```

### Sample Output

For the Alvarez archetype:

> **T. Imai** emerges as the closest match with a 96.5% similarity. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's key traits of creating danger through intelligent movement.
>
> The key similarity lies in movement patterns. Imai consistently finds space away from defenders, a hallmark of the Alvarez archetype.
>
> For development potential, **K. Bos** at 23 offers interesting upside with similar separation values and emerging danger rate.

### Configuration

Available in `src/utils/ai_insights.py`:

```python
# position-specific metric configs
POSITION_METRICS = {
    "forward": {
        "metrics": ["danger_rate", "central_pct", "avg_separation", ...],
        "count_field": "total_entries",
        "criteria": "Forwards are valued for creating danger...",
    },
    # defender and goalkeeper configs...
}
```

### Privacy Considerations

Only aggregated metrics and player names are sent to the AI. No raw tracking coordinates or identifiers beyond public player names are transmitted. All inference is performed on GitHub Models or HuggingFace testbed APIs - data is not used for model training.
