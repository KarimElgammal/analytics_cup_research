---
title: Archetype Comparison Tool
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# Finding Alvarez (and Others) in the A-League

SkillCorner X PySport Analytics Cup 2026 - Research Track

A player similarity study using broadcast tracking data to identify A-League players with characteristics matching world-class archetypes.

**[View Documentation](https://karimelgammal.github.io/analytics_cup_research/)** | **[Live Demo](https://huggingface.co/spaces/KarimElgammal/analytics-cup-research)**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KarimElgammal/analytics_cup_research/HEAD?labpath=submission.ipynb)

---

## Abstract

This research demonstrates how SkillCorner tracking data can identify A-League players matching specific player archetypes. Using data from 10 A-League matches, I built position-specific profiles and computed similarity scores against archetypes derived from StatsBomb World Cup 2022 data.

The system supports three position types:
- **Forwards** (6 archetypes): Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri
- **Defenders** (3 archetypes): Gvardiol, Romero, Hakimi
- **Goalkeepers** (3 archetypes): Lloris, Livakovic, Bounou

Each position uses different event data: forwards use final third entries, defenders use on-ball engagements, and goalkeepers use distribution events. GradientBoosting classifiers calibrated the similarity weights (Forwards AUC 0.656, Defenders AUC 0.845, Goalkeepers AUC 0.993).

Top candidates include Z. Clough (Adelaide) matching Alvarez's movement-focused style, L. Rose matching Gvardiol's ball-playing CB profile, and M. Sutton (Western United) matching Lloris's sweeper-keeper distribution.

---

## Installation

### Option 1: Quick Start (Recommended)

The app launcher handles everything automatically:

```bash
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
./run.sh
```

This will:
1. Install `uv` if not present
2. Create a Python 3.12 virtual environment
3. Install all dependencies
4. Launch the Streamlit app

### Option 2: Manual Setup

```bash
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research

# Using uv (recommended)
uv venv --python 3.12
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

### Interactive App (Archetype Comparison)

```bash
./run.sh
# Or manually:
uv run streamlit run app.py
```

### Research Notebook

```bash
uv run jupyter notebook submission.ipynb
# Or with pip:
jupyter notebook submission.ipynb
```

---

## Archetype Comparison Tool

The interactive Streamlit app (`app.py`) allows comparison across all positions:

| Position | Data Source | Players | Archetypes |
|----------|-------------|---------|------------|
| Forwards | Final third entries (245) | 31 | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri |
| Defenders | Defensive engagements (8,911) | 185 | Gvardiol, Romero, Hakimi |
| Goalkeepers | Distribution events (522) | 13 | Lloris, Livakovic, Bounou |

---

## Project Structure

```
analytics_cup_research/
├── submission.ipynb              # Research notebook (main deliverable)
├── app.py                        # Streamlit app
├── run.sh                        # Self-contained app launcher
├── requirements.txt              # Dependencies
├── docs/                         # MkDocs documentation
└── src/
    ├── core/                     # Archetype, Similarity engine
    ├── data/                     # Data loading
    ├── analysis/                 # Entry, defender, goalkeeper profiles
    ├── statsbomb/                # StatsBomb archetype factory
    └── visualization/            # Plotting functions
```

For detailed technical documentation, see [docs/methodology.md](docs/methodology.md) or visit the [full documentation](https://karimelgammal.github.io/analytics_cup_research/).

---

## AI Insights (Optional)

The app includes AI-powered scouting insights. To enable:

### Local Development

```bash
# Option 1: GitHub Models (recommended)
echo "your_github_token" > github_token.txt

# Option 2: HuggingFace
echo "your_hf_token" > hf_token.txt
```

### HuggingFace Spaces Deployment

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select Streamlit SDK
3. Push this repo to the Space
4. Add secrets in Space Settings:
   - `GITHUB_TOKEN` - GitHub token (recommended, better models)
   - Or `HF_TOKEN` - HuggingFace token

Both work on HF Spaces via environment variables. Token files are gitignored - users never see your tokens.

**Available Models:**
| Backend | Models |
|---------|--------|
| GitHub | Phi-4, GPT-4o Mini |
| HuggingFace | Llama 3.1 8B, Llama 3.2 3B, Qwen 2.5 7B, SmolLM3 3B, Gemma 2 2B |

---

SkillCorner X PySport Analytics Cup 2026 - Research Track Submission
