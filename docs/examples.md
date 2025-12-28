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
| 1 | T. Imai | 28 | Western United | 95.7% | 40.0% | 5 |
| 2 | T. Payne | 22 | Wellington Phoenix | 93.4% | 25.0% | 4 |
| 3 | K. Bos | 23 | Melbourne Victory | 92.3% | 16.7% | 6 |
| 4 | Z. Clough | 30 | Adelaide United | 90.0% | 50.0% | 4 |
| 5 | N. Atkinson | 26 | Melbourne City | 88.4% | 16.7% | 6 |

### Similarity Rankings

![Similarity Rankings](assets/similarity_rankings.png)

**T. Imai** from Western United emerges as the closest match with a 95.7% similarity score. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's playing style of creating through intelligent movement rather than dribbling.

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
#  'gvardiol', 'vandijk', 'hakimi', 'neuer', 'lloris', 'bounou']

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

Generate position-aware AI recommendations with enhanced analysis:

```python
from src.utils.ai_insights import (
    generate_similarity_insight,
    has_valid_token,
    PlayerAnalyzer,
)

# check token exists
if has_valid_token():
    # forward analysis with enhanced insights
    insight = generate_similarity_insight(
        forward_rankings,
        Archetype.from_statsbomb("alvarez"),
        top_n=5,
        position_type="forward",
    )
    print(insight)

# Use PlayerAnalyzer directly for custom analysis
analyzer = PlayerAnalyzer("forward")
gaps = analyzer.compute_development_gaps(player_metrics, archetype_data)
similar = analyzer.find_similar_players("T. Imai", profiles, metric_keys)
confidence = analyzer.get_confidence_level(sample_size=7)
```

### Enhanced Player Summary Format

The AI receives detailed context for each player:

```
- **T. Imai** (Western United): Age 28, Similarity 95.7%, 5 samples [Medium confidence]
  Metrics: Danger Rate: 40.0% (P100), Central %: 20.0% (P25), Separation: 5.6m (P100)
  Development areas: Danger Rate (increase by 55), Central % (increase by 50)
  Similar profiles: T. Payne, K. Bos, C. Elliott
  Age group: 25+ (15 players in group)
```

### Example AI Output (Forward)

> **T. Imai** from Western United emerges as the closest match with 95.7% similarity [Medium confidence]. His exceptional separation (5.64m, P100) and 40% danger rate mirror Alvarez's key traits of creating through intelligent movement rather than dribbling.
>
> **Development Areas**: Imai's priority gaps are Danger Rate (needs to increase by 55 to match target) and Central % (increase by 50). These high-weight metrics suggest coaching focus on finishing and central positioning could elevate his profile further.
>
> **Alternative Profiles**: T. Payne and K. Bos show similar movement patterns and could serve as backup targets. At 22 and 23, both offer development upside.
>
> **Scouting Recommendation**: Despite medium confidence (5 samples), Imai's P100 separation ranking across all 31 forwards is notable. K. Bos (23, High confidence with 12 samples) may be a safer bet for longer-term development.

### Example AI Output (Defender)

> **N. Paull** leads with 82.4% similarity to Gvardiol [High confidence]. His stop danger rate (68%, P85) and pressing rate (72%, P90) indicate an aggressive defensive approach.
>
> **Development Areas**: Primary gap is ball progression - Paull needs to increase carry distance by 15m to match Gvardiol's ball-playing ability.
>
> Unlike Gvardiol who excels at carrying into midfield, Paull shows a more traditional centre-back profile with stronger tackling numbers.

### Example AI Output (Goalkeeper)

> **H. Devenish-Meares** shows 91.2% similarity to Lloris [Medium confidence] with excellent pass success (87%, P95).
>
> At 22 in the U23 age group, he ranks in the top 10% for distribution accuracy among peers. His preference for short distribution matches the sweeper-keeper archetype.
>
> **Similar profiles**: J. Young and T. Sail offer comparable distribution styles if Devenish-Meares is unavailable.
