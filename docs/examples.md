# Examples

## A-League Analysis Results

This example shows results from analysing 10 A-League matches to find players with Julian Alvarez-like characteristics.

### Data Summary

- **245** final third entries detected
- **31** players with 3+ entries profiled
- **10** matches analysed

### Top 5 Alvarez-like Players

| Rank | Player | Age | Team | Similarity | Danger Rate | Entries |
|------|--------|-----|------|------------|-------------|---------|
| 1 | Z. Clough | 29 | Adelaide United | 88.7% | 50.0% | 4 |
| 2 | Z. Machach | 25 | Melbourne Victory | 84.4% | 42.9% | 7 |
| 3 | T. Payne | 21 | Wellington Phoenix | 83.9% | 25.0% | 4 |
| 4 | G. May | 22 | Auckland FC | 80.7% | 55.6% | 9 |
| 5 | T. Imai | 27 | Western United | 79.8% | 40.0% | 5 |

### Similarity Rankings

![Similarity Rankings](assets/similarity_rankings.png)

**Z. Clough** from Adelaide United emerges as the closest match with an 88.7% similarity score. His 50% danger rate means half his final third entries led to shooting opportunities, mirroring Alvarez's clinical finishing.

### Profile Comparison

![Radar Comparison](assets/radar_comparison.png)

The radar chart compares the top 3 candidates against the Alvarez target profile (dashed orange line). The target emphasises high separation, danger rate, and central positioning while setting a low carry percentage — reflecting that Alvarez creates through movement rather than dribbling.

---

## Code: Finding Alvarez-like Players

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine, ScoutingReport

# Build profiles from SkillCorner data
profiler = PlayerProfiler.from_skillcorner(min_entries=3)

# Compute similarity against Alvarez archetype
engine = SimilarityEngine(Archetype.alvarez())
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)

# Display results
print(f"Analysed {len(profiler.entries)} entries from {len(profiler.profiles)} players")
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} ({row.get('team_name', '')}) - {row['similarity_score']}%")

# Generate AI scouting report (requires GitHub token)
report = ScoutingReport(engine)
if report.has_valid_token():
    print(report.generate(top_n=5))
```

---

## Code: Custom Archetype

Create your own archetype for different player types:

```python
from src.core import Archetype, SimilarityEngine

# Define a "pressing forward" archetype
pressing_forward = Archetype.custom(
    "pressing_forward",
    "High-energy forward who creates through pressing and quick transitions"
)

# Set features: speed and counter-attacks matter most
pressing_forward.set_feature("avg_entry_speed", target=85, weight=0.25, direction=1)
pressing_forward.set_feature("quick_break_pct", target=80, weight=0.20, direction=1)
pressing_forward.set_feature("danger_rate", target=70, weight=0.20, direction=1)
pressing_forward.set_feature("avg_defensive_line_dist", target=30, weight=0.15, direction=-1)
pressing_forward.set_feature("avg_separation", target=60, weight=0.10, direction=1)
pressing_forward.set_feature("central_pct", target=50, weight=0.10, direction=1)

# Validate weights sum to ~1.0
warnings = pressing_forward.validate()
if warnings:
    print("Warnings:", warnings)

# Use it
engine = SimilarityEngine(pressing_forward)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=5)
```

---

## Code: Comparing Archetypes

Find which archetype a player best matches:

```python
from src.core import Archetype, SimilarityEngine

archetypes = {
    "alvarez": Archetype.alvarez(),
    "pressing_forward": pressing_forward,
}

player_name = "Z. Clough"

print(f"Archetype fit for {player_name}:")
for name, archetype in archetypes.items():
    engine = SimilarityEngine(archetype)
    engine.fit(profiler.profiles)
    explanation = engine.explain(player_name)
    print(f"  {name}: {explanation['similarity_score']:.1f}%")
```
