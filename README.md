# Finding Alvarez in the A-League

SkillCorner X PySport Analytics Cup 2026 — Research Track

A player similarity study using broadcast tracking data to identify A-League players with Julian Alvarez-like characteristics.

**[View Documentation](https://karimelgammal.github.io/analytics_cup_research/)**

---

## Abstract

Julian Alvarez represents a modern forward archetype characterised by intelligent movement, spatial awareness, and clinical finishing. This research asks whether SkillCorner tracking data can identify A-League players with similar characteristics.

I derived the Alvarez archetype from StatsBomb free event data covering World Cup and Copa América matches. Key metrics include 60% shot accuracy, 20% conversion rate, 24 box touches, and notably only 50% dribble success — indicating he creates danger through movement and positioning rather than dribbling.

Using SkillCorner's Game Intelligence data from 10 A-League matches, I detected 245 final third entries and built player profiles based on spatial positioning, zone preferences, and outcome rates. To calibrate the similarity weights, I trained a GradientBoosting classifier to predict dangerous entries, achieving a cross-validated AUC of 0.656. The feature importances informed weight selection, with separation, entry speed, and defensive line distance emerging as the strongest predictors.

The analysis identified several A-League players exhibiting Alvarez-like characteristics. Top candidates share high danger rates (entries frequently leading to shots), good separation values (finding space between defensive lines), and central zone preferences matching Alvarez's comfort in dangerous areas.

Limitations include the small sample size of 10 matches, the cross-dataset mapping from StatsBomb events to SkillCorner tracking, and the absence of position labels. Despite these constraints, this approach demonstrates that tracking data can support archetype-based player identification for scouting and recruitment purposes.

---

## Quick Start

### Option 1: Using uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research

# Create environment and install dependencies
uv venv --python 3.12
uv pip install -r requirements.txt

# Run the notebook
uv run jupyter notebook submission.ipynb
```

### Option 2: Using pip

```bash
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook submission.ipynb
```

---

## Usage

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine

# Build player profiles from SkillCorner data
profiler = PlayerProfiler.from_skillcorner(min_entries=3)

# Compute similarity against Alvarez archetype
engine = SimilarityEngine(Archetype.alvarez())
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)

# Display results
for row in rankings.to_dicts():
    print(f"{row['rank']}. {row['player_name']} - {row['similarity_score']}%")
```

---

## Project Structure

```
analytics_cup_research/
├── submission.ipynb              # Research notebook
├── requirements.txt              # Dependencies
├── mkdocs.yml                    # Documentation config
├── docs/                         # Documentation
│   ├── index.md
│   ├── getting-started.md
│   ├── examples.md
│   ├── methodology.md            # Technical derivation
│   ├── guide/                    # User guides
│   └── api/                      # API reference
└── src/
    ├── core/                     # Core classes
    │   ├── archetype.py          # Archetype definition
    │   ├── profiler.py           # Player profiling
    │   ├── similarity.py         # Similarity engine
    │   └── report.py             # AI reports
    ├── data/                     # Data loading
    ├── analysis/                 # Analysis functions
    └── visualization/            # Plotting functions
```

For detailed technical documentation, see [docs/methodology.md](docs/methodology.md) or visit the [full documentation](https://karimelgammal.github.io/analytics_cup_research/).

---

SkillCorner X PySport Analytics Cup 2026 — Research Track Submission
