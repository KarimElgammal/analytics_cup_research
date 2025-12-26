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

### Option 1: GitHub Token (Recommended)

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

Both backends work on HF Spaces - add secrets in Space Settings:

- `GITHUB_TOKEN` - GitHub token (recommended)
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

> **Z. Clough** from Adelaide United emerges as the closest match with an 88.7% similarity score. His 50% danger rate indicates clinical finishing similar to Alvarez, though with fewer total entries (4 vs typical 8+).
>
> The key difference between Clough and Alvarez lies in movement patterns. While Clough shows strong separation metrics (8.2m), his central percentage is lower than the archetype target, suggesting he drifts wide more often.
>
> For further observation, **T. Payne** at 22 years old offers the best development potential despite a lower current similarity score. His high entry speed and willingness to attack centrally mirror Alvarez's profile.

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
    player_name="Z. Clough",
    player_profile=player_dict,
    similarity_score=88.7,
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
