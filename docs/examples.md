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
| 1 | T. Imai | 28 | Western United | 96.5% | 40.0% | 5 |
| 2 | N. Atkinson | 26 | Melbourne City | 93.5% | 16.7% | 6 |
| 3 | K. Bos | 23 | Melbourne Victory | 91.6% | 16.7% | 6 |
| 4 | Z. Clough | 30 | Adelaide United | 91.3% | 50.0% | 4 |
| 5 | C. Elliott | 24 | Auckland FC | 90.8% | 33.3% | 3 |

### Similarity Rankings

![Similarity Rankings](assets/similarity_rankings.png)

**T. Imai** from Western United emerges as the closest match with a 96.5% similarity score. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's playing style of creating through intelligent movement rather than dribbling.

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

player_name = "T. Imai"

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

> **T. Imai** from Western United emerges as the closest match with 96.5% similarity. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's key traits of creating danger through intelligent movement rather than dribbling.
>
> The key similarity lies in movement patterns. Imai consistently finds space away from defenders, a hallmark of the Alvarez archetype. His 0% central percentage suggests he operates from wide areas but still generates shooting opportunities.
>
> For development potential, **K. Bos** at 23 offers interesting upside. His similar separation values and emerging danger rate align with Alvarez's profile.

### Example AI Output (Defender)

> **N. Paull** leads the rankings with 82.4% similarity to Gvardiol. His stop danger rate of 68% and pressing rate of 72% suggest a similarly aggressive defensive approach.
>
> Unlike Gvardiol who excels at carrying the ball out, Paull shows lower progression metrics but stronger tackling numbers. This suggests a more traditional centre-back profile.

### Example AI Output (Goalkeeper)

> **H. Devenish-Meares** shows 91.2% similarity to Lloris with excellent pass success (87%) and quick distribution tendency.
>
> At 22, he represents the strongest development prospect among the top candidates. His preference for short distribution matches the sweeper-keeper archetype.
