# Examples

## Finding Alvarez-like Players

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine, ScoutingReport

# Build profiles
profiler = PlayerProfiler.from_skillcorner(min_entries=3)

# Compute similarity
engine = SimilarityEngine(Archetype.alvarez())
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']}%")
```

## Custom Archetype

```python
from src.core import Archetype

pressing_forward = Archetype.custom("pressing_forward", "High-energy forward")
pressing_forward.set_feature("avg_entry_speed", target=85, weight=0.25)
pressing_forward.set_feature("quick_break_pct", target=80, weight=0.20)
pressing_forward.set_feature("danger_rate", target=70, weight=0.20)
```
