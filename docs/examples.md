# Examples

## A-League Analysis Results

This example shows results from analysing 10 A-League matches using 12 pre-built archetypes across forwards, defenders, and goalkeepers.

### Data Summary

- **245** final third entries detected (forwards)
- **8,911** defensive engagements detected (defenders)
- **522** goalkeeper distributions detected
- **10** matches analysed

### Top 5 Alvarez-like Players (Forwards)

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

The radar chart compares the top 3 candidates against the Alvarez target profile (dashed orange line). The target emphasises high separation, danger rate, and central positioning while setting a low carry percentage, reflecting that Alvarez creates through movement rather than dribbling.

---

## Code: Finding Similar Players

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# See all 12 available archetypes
print(Archetype.list_available())
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'romero', 'hakimi', 'lloris', 'livakovic', 'bounou']

# Load any archetype from StatsBomb World Cup 2022 data
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual stats

# Compute similarity against player profiles
engine = SimilarityEngine(archetype)
engine.fit(profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']:.1f}%")
```

---

## Code: Multi-Position Analysis

Analyse players across different positions:

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# Forward analysis
forward_archetype = Archetype.from_statsbomb("giroud")  # Target man
engine = SimilarityEngine(forward_archetype)
engine.fit(forward_profiles)
forward_rankings = engine.rank(top_n=5)

# Defender analysis
defender_archetype = Archetype.from_statsbomb("hakimi")  # Attacking wing-back
engine = SimilarityEngine(defender_archetype)
engine.fit(defender_profiles)
defender_rankings = engine.rank(top_n=5)

# Goalkeeper analysis
gk_archetype = Archetype.from_statsbomb("lloris")  # Sweeper-keeper
engine = SimilarityEngine(gk_archetype)
engine.fit(goalkeeper_profiles)
gk_rankings = engine.rank(top_n=5)
```

---

## Code: Custom Archetype

Create your own archetype for different player types:

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

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
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# Compare against multiple archetypes
archetypes = {
    "alvarez": Archetype.from_statsbomb("alvarez"),
    "giroud": Archetype.from_statsbomb("giroud"),
    "rashford": Archetype.from_statsbomb("rashford"),
}

player_name = "Z. Clough"

print(f"Archetype fit for {player_name}:")
for name, archetype in archetypes.items():
    engine = SimilarityEngine(archetype)
    engine.fit(profiles)
    explanation = engine.explain(player_name)
    print(f"  {name}: {explanation['similarity_score']:.1f}%")
```

---

## Code: AI Scouting Insights

Generate position-aware AI recommendations:

```python
from src.utils.ai_insights import generate_similarity_insight, has_valid_token

# check token exists
if has_valid_token():
    # forward analysis
    insight = generate_similarity_insight(
        forward_rankings,
        Archetype.from_statsbomb("alvarez"),
        top_n=5,
        position_type="forward",
    )
    print(insight)

    # defender analysis
    insight = generate_similarity_insight(
        defender_rankings,
        defender_archetype,
        top_n=5,
        position_type="defender",
    )
    print(insight)
```

### Example AI Output (Forward)

> **Z. Clough** from Adelaide United emerges as the closest match with 88.7% similarity. His 50% danger rate mirrors Alvarez's clinical finishing, though with fewer entries (4 vs typical 8+).
>
> The key difference between Clough and the archetype lies in movement patterns. While he shows strong separation (8.2m), his central percentage is lower than target.
>
> For development potential, **T. Payne** at 22 offers interesting upside. His entry speed and willingness to attack centrally align with Alvarez's profile.

### Example AI Output (Defender)

> **N. Paull** leads the rankings with 82.4% similarity to Gvardiol. His stop danger rate of 68% and pressing rate of 72% suggest a similarly aggressive defensive approach.
>
> Unlike Gvardiol who excels at carrying the ball out, Paull shows lower progression metrics but stronger tackling numbers. This suggests a more traditional centre-back profile.

### Example AI Output (Goalkeeper)

> **H. Devenish-Meares** shows 91.2% similarity to Lloris with excellent pass success (87%) and quick distribution tendency.
>
> At 22, he represents the strongest development prospect among the top candidates. His preference for short distribution matches the sweeper-keeper archetype.
