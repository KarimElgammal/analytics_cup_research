# Getting Started

## Installation

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then set up the project:

```bash
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

## Data Setup

Place your SkillCorner tracking data in the `data/` directory:

```
data/
├── match_1234/
│   ├── match_data.json
│   └── dynamic_events.csv
└── match_5678/
    └── ...
```

## Basic Usage

```python
from src.data.loader import load_all_events, add_team_names
from src.analysis.entries import detect_entries, classify_entries
from src.analysis.profiles import build_player_profiles, filter_profiles
from src.core.archetype import Archetype
from src.core.similarity import SimilarityEngine

# Load and process SkillCorner data
events = load_all_events()
events = add_team_names(events)
entries = detect_entries(events)
entries = classify_entries(entries)
profiles = build_player_profiles(entries)
profiles = filter_profiles(profiles, min_entries=3)

# Load archetype from StatsBomb data (10 available)
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

Load any of these 10 pre-built archetypes:

```python
# List all available
Archetype.list_available()
# ['alvarez', 'giroud', 'kane', 'lewandowski', 'rashford', 'en_nesyri',
#  'gvardiol', 'romero', 'lloris', 'livakovic']

# Load specific archetype
archetype = Archetype.from_statsbomb("giroud")  # Target man style
archetype = Archetype.from_statsbomb("rashford")  # Pace/dribbling style
archetype = Archetype.from_statsbomb("gvardiol")  # Ball-playing defender
```

## Running the Notebook

```bash
uv run jupyter notebook submission.ipynb
# Or with pip:
jupyter notebook submission.ipynb
```
