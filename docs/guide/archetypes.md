# Defining Archetypes

An archetype defines the target player profile you want to match.

## Pre-built Archetypes (18 Available)

Load any of the 18 pre-built archetypes from StatsBomb World Cup 2022 data:

```python
from src.core import Archetype

# List all available archetypes
print(Archetype.list_available())
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'enzo', 'tchouameni', 'depaul', 'griezmann', 'pedri', 'bellingham',
#  'gvardiol', 'vandijk', 'hakimi', 'neuer', 'lloris', 'bounou']

# Load a specific archetype
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual World Cup 2022 stats
```

### By Position

| Position | Archetypes | A-League Events |
|----------|------------|-----------------|
| **Forwards** | alvarez, giroud, kane, lewandowski, rashford, en_nesyri | 245 entries |
| **Midfielders** | enzo, tchouameni, depaul, griezmann, pedri, bellingham | 12,975 possessions |
| **Defenders** | gvardiol, vandijk, hakimi | 8,911 engagements |
| **Goalkeepers** | neuer, lloris, bounou | 522 distributions |

### Examples

```python
# Forwards
archetype = Archetype.from_statsbomb("alvarez")    # Intelligent movement
archetype = Archetype.from_statsbomb("giroud")     # Target man
archetype = Archetype.from_statsbomb("rashford")   # Pace and dribbling

# Midfielders
archetype = Archetype.from_statsbomb("enzo")       # Box-to-box
archetype = Archetype.from_statsbomb("tchouameni") # Defensive anchor
archetype = Archetype.from_statsbomb("pedri")      # Technical control

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
| `quick_break_pct` | % of entries in quick transitions |

### Midfielder Features
| Feature | Description |
|---------|-------------|
| `progressive_pass_pct` | % of passes advancing play forward |
| `pressing_rate` | Pressing engagements frequency |
| `tackle_success_rate` | % of successful tackles |
| `key_pass_rate` | % of passes leading to shots |
| `central_presence_pct` | % of time in central zones |

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
