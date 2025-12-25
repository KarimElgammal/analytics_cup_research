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
from src.core import Archetype, PlayerProfiler, SimilarityEngine

# Build player profiles from SkillCorner data
profiler = PlayerProfiler(min_entries=3)
profiler.load_data()
profiler.detect_entries()
profiler.build_profiles()

# Use Alvarez archetype and compute similarity
archetype = Archetype.alvarez()
engine = SimilarityEngine(archetype)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']}%")
```

## Running the Notebook

```bash
uv run jupyter notebook submission.ipynb
# Or with pip:
jupyter notebook submission.ipynb
```
