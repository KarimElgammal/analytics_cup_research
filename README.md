# Finding Alvarez in the A-League

SkillCorner X PySport Analytics Cup 2026 — Research Track

A player similarity study using broadcast tracking data to identify A-League players with characteristics matching world-class archetypes.

**[View Documentation](https://karimelgammal.github.io/analytics_cup_research/)**

---

## Abstract

This research demonstrates how SkillCorner tracking data can identify A-League players matching specific player archetypes. Using data from 10 A-League matches, I built position-specific profiles and computed similarity scores against archetypes derived from StatsBomb World Cup 2022 data.

The system supports three position types:
- **Forwards** (6 archetypes): Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri
- **Defenders** (2 archetypes): Gvardiol, Romero
- **Goalkeepers** (3 archetypes): Lloris, Livakovic, Bounou

Each position uses different event data: forwards use final third entries, defenders use on-ball engagements, and goalkeepers use distribution events. A GradientBoosting classifier (AUC 0.656) calibrated the similarity weights for forwards.

Top candidates include Z. Clough (Adelaide) matching Alvarez's movement-focused style, H. Steele (Central Coast) matching Gvardiol's ball-playing CB profile, and M. Sutton (Western United) matching Lloris's sweeper-keeper distribution.

---

## Quick Start

### Run the Interactive App

```bash
# Clone and setup
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
uv venv --python 3.12
uv pip install -r requirements.txt

# Run the archetype comparison tool
./run_archetype_compare.sh
```

### Run the Notebook

```bash
uv run jupyter notebook submission.ipynb
```

---

## Archetype Comparison Tool

The interactive Streamlit app (`archetype_compare.py`) allows comparison across all positions:

| Position | Data Source | Players | Archetypes |
|----------|-------------|---------|------------|
| Forwards | Final third entries (245) | 31 | Alvarez, Giroud, Kane, Lewandowski, Rashford, En-Nesyri |
| Defenders | Defensive engagements (8,911) | 185 | Gvardiol, Romero |
| Goalkeepers | Distribution events (522) | 13 | Lloris, Livakovic, Bounou |

---

## Project Structure

```
analytics_cup_research/
├── submission.ipynb              # Research notebook
├── archetype_compare.py          # Streamlit app
├── run_archetype_compare.sh      # App launcher
├── requirements.txt              # Dependencies
├── docs/                         # Documentation
└── src/
    ├── core/                     # Archetype, Similarity engine
    ├── data/                     # Data loading
    ├── analysis/                 # Entry, defender, goalkeeper profiles
    ├── statsbomb/                # StatsBomb archetype factory
    └── visualization/            # Plotting functions
```

For detailed technical documentation, see [docs/methodology.md](docs/methodology.md) or visit the [full documentation](https://karimelgammal.github.io/analytics_cup_research/).

---

SkillCorner X PySport Analytics Cup 2026 — Research Track Submission
