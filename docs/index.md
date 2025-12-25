# SkillCorner Scout

**Player archetype matching using SkillCorner tracking data**

SkillCorner Scout is a Python library for identifying players who match specific playing styles using broadcast tracking data.

## Features

- **Archetype Definition**: Create custom player archetypes with target profiles
- **Profile Building**: Extract player profiles from SkillCorner tracking data
- **Similarity Scoring**: Rank players by weighted cosine similarity
- **AI Reports**: Generate natural language scouting reports

## Quick Example

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine

# Use the pre-built Alvarez archetype
archetype = Archetype.alvarez()

# Build player profiles
profiler = PlayerProfiler.from_skillcorner(min_entries=3)

# Compute similarity rankings
engine = SimilarityEngine(archetype)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)
```

## Project Origin

Developed for the SkillCorner X PySport Analytics Cup 2026 Research Track.
