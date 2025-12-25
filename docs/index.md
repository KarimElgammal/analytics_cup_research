# SkillCorner Scout

**Player archetype matching using SkillCorner tracking data**

SkillCorner Scout is a Python library for identifying players who match specific playing styles using broadcast tracking data. Define an archetype (like Julian Alvarez's intelligent movement style), and find similar players in any league with SkillCorner coverage.

## Features

- **Archetype Definition**: Create custom player archetypes with target profiles and weighted features
- **Profile Building**: Automatically extract player profiles from SkillCorner tracking data
- **Similarity Scoring**: Rank players by weighted cosine similarity to your target archetype
- **AI Reports**: Generate natural language scouting reports using LLM APIs

## Quick Example

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine

# Use the pre-built Alvarez archetype
archetype = Archetype.alvarez()

# Build player profiles from SkillCorner data
profiler = PlayerProfiler.from_skillcorner(min_entries=3)

# Compute similarity rankings
engine = SimilarityEngine(archetype)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']}%")
```

## A-League Results

Using 10 A-League matches, the analysis identified **Z. Clough** (Adelaide United) as the closest match to the Alvarez archetype with an 88.7% similarity score. See [Examples](examples.md) for full results and figures.

## Project Origin

Developed for the SkillCorner X PySport Analytics Cup 2026 Research Track. The research question: "Can SkillCorner tracking data identify A-League players with Julian Alvarez-like characteristics?"

## Next Steps

- [Getting Started](getting-started.md) - Installation and setup
- [User Guide](guide/archetypes.md) - Learn how to define archetypes
- [Examples](examples.md) - Full A-League results with figures
- [API Reference](api/archetype.md) - Complete API documentation
