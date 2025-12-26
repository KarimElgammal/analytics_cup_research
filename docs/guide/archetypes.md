# Defining Archetypes

An archetype defines the target player profile you want to match.

## Pre-built Archetypes (12 Available)

Load any of the 12 pre-built archetypes from StatsBomb World Cup 2022 data:

```python
from src.core import Archetype

# List all available archetypes
print(Archetype.list_available())
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'romero', 'hakimi', 'lloris', 'livakovic', 'bounou']

# Load a specific archetype
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual World Cup 2022 stats
```

### By Position

| Position | Archetypes | ML Model AUC |
|----------|------------|--------------|
| **Forwards** | alvarez, giroud, kane, lewandowski, rashford, en_nesyri | 0.656 |
| **Defenders** | gvardiol, romero, hakimi | 0.845 |
| **Goalkeepers** | lloris, livakovic, bounou | 0.993 |

### Examples

```python
# Forwards
archetype = Archetype.from_statsbomb("alvarez")    # Intelligent movement
archetype = Archetype.from_statsbomb("giroud")     # Target man
archetype = Archetype.from_statsbomb("rashford")   # Pace and dribbling

# Defenders
archetype = Archetype.from_statsbomb("gvardiol")   # Ball-playing CB
archetype = Archetype.from_statsbomb("hakimi")     # Attacking wing-back

# Goalkeepers
archetype = Archetype.from_statsbomb("lloris")     # Sweeper-keeper
```

## Custom Archetypes

Create your own archetype for specific player types:

```python
archetype = Archetype.custom("target_man", "Aerial presence forward")
archetype.set_feature("danger_rate", target=80, weight=0.25, direction=1)
archetype.set_feature("central_pct", target=90, weight=0.20, direction=1)
```

## Available Features

### Forward Features
| Feature | Description |
|---------|-------------|
| `avg_entry_speed` | Mean speed during entries (m/s) |
| `avg_separation` | Distance from nearest defender (m) |
| `central_pct` | % of entries through central zone |
| `danger_rate` | % of entries leading to shots |
| `carry_pct` | % of entries via carries |
| `quick_break_pct` | % of entries in quick transitions |

### Defender Features
| Feature | Description |
|---------|-------------|
| `stop_danger_rate` | % of engagements stopping dangerous attacks |
| `avg_engagement_distance` | Mean distance from goal during engagements |
| `reduce_danger_rate` | % of engagements reducing attack danger |
| `beaten_by_possession_rate` | % of times beaten by ball carrier |

### Goalkeeper Features
| Feature | Description |
|---------|-------------|
| `pass_success_rate` | % of successful distributions |
| `avg_pass_distance` | Mean distribution distance |
| `long_pass_pct` | % of long distributions |
| `under_pressure_pct` | % of distributions under pressure |
