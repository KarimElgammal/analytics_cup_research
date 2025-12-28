# AI Reports

Generate AI-powered scouting insights comparing A-League players to world-class archetypes.

## Overview

The AI reports system supports multiple backends to generate contextual scouting recommendations:

| Backend | Models | Setup | Budget |
|---------|--------|-------|--------|
| **GitHub Models** | Phi-4, GPT-4o Mini | `github_token.txt` or `GITHUB_TOKEN` env | $2/month |
| **HuggingFace** | Llama 3.1/3.2, Qwen 2.5, SmolLM3, Gemma 2 | `hf_token.txt` or `HF_TOKEN` env | $2/month |

Reports are position-aware and enriched with dataset statistics.

## Setup

### Option 1: GitHub Token

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a new token with default permissions
3. Save it to `github_token.txt` in the project root

```bash
export GITHUB_TOKEN=your_token_here
```

### Option 2: HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a token with Inference Providers permission
3. Save it to `hf_token.txt` in the project root

```bash
export HF_TOKEN=your_token_here
```

### HuggingFace Spaces Deployment

Both backends work on HF Spaces by adding secrets in Space Settings:

- `GITHUB_TOKEN` - GitHub token
- `HF_TOKEN` - HuggingFace token

Token files are gitignored, so users never see your credentials.

### Basic Usage

```python
from src.utils.ai_insights import (
    generate_similarity_insight,
    generate_player_report,
    has_valid_token,
)

# check token exists
if has_valid_token():
    insight = generate_similarity_insight(
        ranked_players,
        archetype,
        top_n=5,
        position_type="forward",
    )
    print(insight)
```

## Position-Aware Insights

The AI receives different context based on position type. Each position has specific metrics that matter most for scouting.

### Forward Metrics

- Danger Rate: percentage of entries leading to shots
- Central %: how often player operates centrally
- Separation: distance from nearest defender
- Entry Speed: pace entering final third
- Depth: distance from defensive line (lower = closer to goal)
- Quick Break %: percentage of entries during counter-attacks

### Defender Metrics

- Stop Danger Rate: percentage of engagements stopping danger
- Reduce Danger Rate: engagements reducing but not stopping danger
- Pressing Rate: proactive engagement tendency
- Goal Side Rate: maintaining goal-side position
- Engagement Distance: how far from goal when engaging

### Goalkeeper Metrics

- Pass Success Rate: distribution accuracy
- Pass Distance: average distribution length
- Long Pass %: direct distribution tendency
- Quick Distribution %: speed of restarts

## What Gets Sent to the AI

The prompt includes enriched context for accurate insights:

1. **Position type** and archetype name
2. **Dataset size** (events and players)
3. **Feature weights** showing what matters most (e.g., `avg_separation: 23%`, `danger_rate: 22%`)
4. **Archetype targets** with ideal values from StatsBomb (e.g., `danger_rate: 95`, `central_pct: 70`)
5. **Top candidates** with metrics and **percentile ranks** vs full dataset (e.g., `40.0% (P85)`)
6. **Confidence levels** for each player: High (10+), Medium (5-9), Low (<5 samples)
7. **Development gaps** showing priority areas to improve (metric, direction, gap size, weight)
8. **Similar players** in the dataset with comparable profiles
9. **Age group context** (U21, U23, U25, 25+) and group size
10. **Dataset statistics** (mean ± std for each metric)
11. **Domain knowledge request** asking the model to incorporate knowledge about players, teams, and leagues

No raw tracking data or player identifiers beyond names are sent.

## Enhanced Analysis Features

The AI insights module provides several advanced analysis capabilities:

### Confidence Levels

Based on sample size, each player receives a confidence rating:

| Level | Samples | Meaning |
|-------|---------|---------|
| 🟢 High | 10+ | Reliable profile based on sufficient data |
| 🟡 Medium | 5-9 | Reasonable estimate, more data would help |
| 🔴 Low | <5 | Preliminary profile, needs more observations |

### Development Gaps

For each player, the system identifies which metrics differ most from the archetype target:

```
Development areas: Danger Rate (increase by 55), Central % (increase by 50)
```

Gaps are prioritised by: `gap_size × metric_weight`, so high-weight metrics with large gaps are flagged first.

### Similar Players

The system finds other players in the dataset with similar profiles using cosine similarity on z-score normalised metrics:

```
Similar profiles: T. Payne, K. Bos, C. Elliott
```

This helps scouts identify backup candidates with comparable styles.

### Age Group Percentiles

Players are compared within their age group (U21, U23, U25, 25+) to contextualise their development stage.

## Domain Knowledge

The AI is encouraged to use its training knowledge to enrich insights:

- **Player context**: Career history, international caps, previous clubs, known strengths
- **Team context**: Club reputation, playing style, academy quality
- **League context**: A-League level compared to other leagues, transfer market dynamics
- **Archetype context**: The reference player's (e.g., Alvarez) actual playing style and career

This adds valuable context beyond what the tracking data alone can show.

## Example Output

For a forward archetype like Alvarez:

> **T. Imai** from Western United emerges as the closest match with a 95.7% similarity score. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's key traits of creating danger through intelligent movement rather than dribbling.
>
> The key similarity lies in movement patterns. Imai consistently finds space away from defenders, a hallmark of the Alvarez archetype. His 0% central percentage suggests he operates from wide areas but still generates shooting opportunities.
>
> For development potential, **K. Bos** at 23 years old offers interesting upside with similar separation values and emerging danger rate that align with Alvarez's profile.

## Customisation

### Change Model

```python
# Available models:
# GitHub: "phi-4", "gpt-4o-mini"
# HuggingFace: "llama-3.1-8b", "llama-3.2-3b", "qwen-2.5-7b", "smollm3-3b", "gemma-2-2b"

insight = generate_similarity_insight(
    ranked,
    archetype,
    model="llama-3.1-8b",  # or any model above
    position_type="forward",
)
```

### Individual Player Reports

```python
from src.utils.ai_insights import generate_player_report

report = generate_player_report(
    player_name="T. Imai",
    player_profile=player_dict,
    similarity_score=95.7,
    archetype=archetype,
    position_type="forward",
)
```

## Streamlit Integration

The interactive app includes AI insights in a dedicated tab:

```python
# in app.py
if has_valid_token():
    with st.spinner("Generating AI scouting insight..."):
        insight = generate_similarity_insight(
            results, archetype, top_n=5, position_type=position_type
        )
    st.markdown(insight)
```
