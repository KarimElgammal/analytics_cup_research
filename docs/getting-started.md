# Getting Started

## Try Without Installing

No setup required - try the project directly in your browser:

[Live Streamlit Demo](https://huggingface.co/spaces/KarimElgammal/analytics-cup-research){ .md-button .md-button--primary }
[Open Notebook in Binder](https://mybinder.org/v2/gh/KarimElgammal/analytics_cup_research/HEAD?labpath=submission.ipynb){ .md-button }

---

## Quick Start (Recommended)

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

## Manual Installation

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

### Interactive App

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

## Basic Usage

```python
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine
from src.data.loader import load_all_events, add_team_names
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles

# Load and process SkillCorner data
events = load_all_events()
events = add_team_names(events)
entries = detect_entries(events)
entries = classify_entries(entries)
profiles = build_player_profiles(entries)
profiles = filter_profiles(profiles, min_entries=3)

# Load archetype from StatsBomb data (12 available)
archetype = Archetype.from_statsbomb("alvarez")
print(archetype.description)  # Shows actual stats

# Compute similarity rankings
engine = SimilarityEngine(archetype)
engine.fit(profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']:.1f}%")
```

## Available Archetypes

Load any of these 12 pre-built archetypes:

```python
# List all available
Archetype.list_available()
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'vandijk', 'hakimi', 'neuer', 'lloris', 'bounou']

# Load specific archetype
archetype = Archetype.from_statsbomb("giroud")   # Target man style
archetype = Archetype.from_statsbomb("rashford") # Pace/dribbling style
archetype = Archetype.from_statsbomb("gvardiol") # Ball-playing CB
archetype = Archetype.from_statsbomb("hakimi")   # Attacking wing-back
```
