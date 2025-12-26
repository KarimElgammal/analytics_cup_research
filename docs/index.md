# Finding Alvarez (and Others) in the A-League

**Player archetype matching using SkillCorner tracking data**

A player similarity study for identifying A-League players who match world-class archetypes. Define an archetype (like Julian Alvarez's intelligent movement style), and find similar players using SkillCorner broadcast tracking data.

## Features

- **12 Pre-built Archetypes**: Computed from real StatsBomb World Cup 2022 data
- **ML-Calibrated Weights**: GradientBoosting models for each position type
- **Profile Building**: Automatically extract player profiles from SkillCorner tracking data
- **Similarity Scoring**: Rank players by weighted cosine similarity
- **AI Scouting Insights**: Position-aware LLM recommendations via GitHub Models API
- **Interactive App**: Streamlit-based comparison tool with radar charts and AI analysis

## Available Archetypes

| Position | Players | ML Model AUC |
|----------|---------|--------------|
| **Forward** | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri | 0.656 |
| **Defender** | Gvardiol, Romero, Hakimi | 0.845 |
| **Goalkeeper** | Lloris, Livakovic, Bounou | 0.993 |

## Quick Start

```bash
# Clone and run (handles everything automatically)
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
./run.sh
```

## Quick Example

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# Load archetype from StatsBomb data (12 players available)
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual World Cup 2022 stats

# See all available archetypes
print(Archetype.list_available())
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'romero', 'hakimi', 'lloris', 'livakovic', 'bounou']

# Compute similarity rankings against your player profiles
engine = SimilarityEngine(archetype)
engine.fit(profiles)  # Your SkillCorner player profiles
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']:.1f}%")
```

## A-League Results

Using 10 A-League matches, the analysis identified **Z. Clough** (Adelaide United) as the closest match to the Alvarez archetype with an 87.9% similarity score. See [Examples](examples.md) for full results and figures.

## Project Origin

Developed for the SkillCorner X PySport Analytics Cup 2026 Research Track. The research question: "Can SkillCorner tracking data identify A-League players with specific archetype characteristics?"

## Next Steps

- [Getting Started](getting-started.md) - Installation and setup
- [Methodology](methodology.md) - Full technical documentation with ML model details
- [User Guide](guide/archetypes.md) - Learn how to define archetypes
- [AI Reports](guide/reports.md) - Position-aware AI scouting insights
- [Examples](examples.md) - Full A-League results with figures
- [API Reference](api/archetype.md) - Complete API documentation
