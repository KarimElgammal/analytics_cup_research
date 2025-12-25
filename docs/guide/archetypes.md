# Defining Archetypes

An archetype defines the target player profile you want to match.

## Pre-built Archetypes

```python
from src.core import Archetype

archetype = Archetype.alvarez()
```

## Custom Archetypes

```python
archetype = Archetype.custom("target_man", "Aerial presence forward")
archetype.set_feature("danger_rate", target=80, weight=0.25, direction=1)
archetype.set_feature("central_pct", target=90, weight=0.20, direction=1)
```

## Available Features

| Feature | Description |
|---------|-------------|
| `avg_entry_speed` | Mean speed during entries (m/s) |
| `avg_separation` | Distance from nearest defender (m) |
| `central_pct` | % of entries through central zone |
| `danger_rate` | % of entries leading to shots |
