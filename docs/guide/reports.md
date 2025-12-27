# AI Reports

Generate AI-powered scouting insights comparing A-League players to world-class archetypes.

## Overview

The AI reports system supports multiple backends to generate contextual scouting recommendations:

| Backend | Models | Setup | Budget |
|---------|--------|-------|--------|
| **GitHub Models** | Phi-4, GPT-4o Mini | `github_token.txt` or `GITHUB_TOKEN` env | $2/month |
| **HuggingFace** | Llama 3.1/3.2, Qwen 2.5, SmolLM3, Gemma 2 | `hf_token.txt` or `HF_TOKEN` env | $2/month |

Reports are position-aware and enriched with ML model confidence scores and dataset statistics.

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
- Carry %: dribbling vs passing entries

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

The prompt includes:

1. Position type and archetype name
2. Dataset size (events and players)
3. ML model AUC score (reliability indicator)
4. Position-specific evaluation criteria
5. Top 5 candidates with their metrics
6. Dataset averages for comparison

No raw tracking data or player identifiers beyond names are sent.

## Example Output

For a forward archetype like Alvarez:

> **T. Imai** from Western United emerges as the closest match with a 96.5% similarity score. His exceptional separation (5.64m) and 40% danger rate mirror Alvarez's key traits of creating danger through intelligent movement rather than dribbling.
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
    similarity_score=96.5,
    archetype=archetype,
    position_type="forward",
)
```

## ML Model Context

The AI receives the ML model's AUC score to calibrate confidence:

| Position | AUC | Interpretation |
|----------|-----|----------------|
| Forward | 0.656 | Moderate reliability, more uncertainty |
| Defender | 0.845 | Good reliability |
| Goalkeeper | 0.993 | Excellent reliability |

Higher AUC means the underlying similarity rankings are more trustworthy, which the AI factors into its recommendations.

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
