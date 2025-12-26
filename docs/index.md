# SkillCorner Scout

**Player archetype matching using SkillCorner tracking data**

SkillCorner Scout is a Python library for identifying players who match specific playing styles using broadcast tracking data. Define an archetype (like Julian Alvarez's intelligent movement style), and find similar players in any league with SkillCorner coverage.

## Features

- **11 Pre-built Archetypes**: Computed from real StatsBomb World Cup 2022 data
- **Profile Building**: Automatically extract player profiles from SkillCorner tracking data
- **Similarity Scoring**: Rank players by ML-calibrated weighted cosine similarity
- **AI Reports**: Generate natural language scouting reports using LLM APIs
- **Custom Archetypes**: Create your own archetypes with target profiles

## Available Archetypes

| Position | Players |
|----------|---------|
| **Forward** | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri |
| **Defender** | Gvardiol, Romero |
| **Goalkeeper** | Lloris, Livakovic, Bounou |

## Quick Example

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# Load archetype from StatsBomb data (10 players available)
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual World Cup 2022 stats

# See all available archetypes
print(Archetype.list_available())
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'romero', 'lloris', 'livakovic', 'bounou']

# Compute similarity rankings against your player profiles
engine = SimilarityEngine(archetype)
engine.fit(profiles)  # Your SkillCorner player profiles
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']:.1f}%")
```

## A-League Results

Using 10 A-League matches, the analysis identified **Z. Clough** (Adelaide United) as the closest match to the Alvarez archetype with an 88.7% similarity score. See [Examples](examples.md) for full results and figures.

## Project Origin

Developed for the SkillCorner X PySport Analytics Cup 2026 Research Track. The research question: "Can SkillCorner tracking data identify A-League players with Julian Alvarez-like characteristics?"

## Next Steps

- [Getting Started](getting-started.md) - Installation and setup
- [Methodology](methodology.md) - Full technical documentation with player selection rationale
- [User Guide](guide/archetypes.md) - Learn how to define archetypes
- [Examples](examples.md) - Full A-League results with figures
- [API Reference](api/archetype.md) - Complete API documentation
