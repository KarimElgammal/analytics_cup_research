# Finding Alvarez (and Others) in the A-League

**Player archetype matching using SkillCorner tracking data**

A player similarity study for identifying A-League players who match world-class archetypes. Define an archetype (like Julian Alvarez's intelligent movement style), and find similar players using SkillCorner broadcast tracking data.

[Live Demo](https://huggingface.co/spaces/KarimElgammal/analytics-cup-research){ .md-button .md-button--primary }
[GitHub](https://github.com/KarimElgammal/analytics_cup_research){ .md-button }
[Submission Notebook](https://github.com/KarimElgammal/analytics_cup_research/blob/main/submission.ipynb){ .md-button }

## Features

- **12 Pre-built Archetypes**: Computed from StatsBomb World Cup 2022 data
- **Correlation-based Weights**: Feature weights derived from A-League data analysis
- **Profile Building**: Automatically extract player profiles from SkillCorner tracking data
- **Similarity Scoring**: Rank players by weighted cosine similarity
- **AI Scouting Insights**: Statistics-aware LLM analysis via GitHub Models or HuggingFace
- **Interactive App**: Streamlit-based comparison tool with radar charts and AI analysis

## Available Archetypes

| Position | Players | Events |
|----------|---------|--------|
| **Forward** | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri | 245 entries |
| **Defender** | Gvardiol, Van Dijk, Hakimi | 8,911 engagements |
| **Goalkeeper** | Neuer, Lloris, Bounou | 522 distributions |

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
#  'gvardiol', 'vandijk', 'hakimi', 'neuer', 'lloris', 'bounou']

# Compute similarity rankings against your player profiles
engine = SimilarityEngine(archetype)
engine.fit(profiles)  # Your SkillCorner player profiles
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']:.1f}%")
```

## A-League Results

Using 10 A-League matches, the analysis identified **T. Imai** (Western United) as the closest match to the Alvarez archetype with a 95.7% similarity score. See [Examples](examples.md) for full results and figures.

## Project Origin

Developed for the SkillCorner X PySport Analytics Cup 2026 Research Track. The research question: "Can SkillCorner tracking data identify A-League players with specific archetype characteristics?"

## Next Steps

- [Getting Started](getting-started.md) - Installation and setup
- [Methodology](methodology.md) - Full technical documentation
- [User Guide](guide/archetypes.md) - Learn how to define archetypes
- [AI Reports](guide/reports.md) - Position-aware AI scouting insights
- [Examples](examples.md) - Full A-League results with figures
- [API Reference](api/archetype.md) - Complete API documentation
