# Finding Alvarez in the A-League

SkillCorner X PySport Analytics Cup 2026 — Research Track

A player similarity study using broadcast tracking data to identify A-League players with Julian Alvarez-like characteristics.

---

## Abstract

Julian Alvarez represents a modern forward archetype characterised by intelligent movement, spatial awareness, and clinical finishing. This research asks whether SkillCorner tracking data can identify A-League players with similar characteristics.

I derived the Alvarez archetype from StatsBomb free event data covering World Cup and Copa América matches. Key metrics include 60% shot accuracy, 20% conversion rate, 24 box touches, and notably only 50% dribble success — indicating he creates danger through movement and positioning rather than dribbling.

Using SkillCorner's Game Intelligence data from 10 A-League matches, I detected 245 final third entries and built player profiles based on spatial positioning, zone preferences, and outcome rates. To calibrate the similarity weights, I trained a GradientBoosting classifier to predict dangerous entries, achieving a cross-validated AUC of 0.656. The feature importances informed weight selection, with separation, entry speed, and defensive line distance emerging as the strongest predictors.

The analysis identified several A-League players exhibiting Alvarez-like characteristics. Top candidates share high danger rates (entries frequently leading to shots), good separation values (finding space between defensive lines), and central zone preferences matching Alvarez's comfort in dangerous areas.

Limitations include the small sample size of 10 matches, the cross-dataset mapping from StatsBomb events to SkillCorner tracking, and the absence of position labels. Despite these constraints, this approach demonstrates that tracking data can support archetype-based player identification for scouting and recruitment purposes.

---

## Run Instructions

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/analytics_cup_research.git
cd analytics_cup_research

# Run with uv
uv venv --python 3.12
uv pip install -r requirements.txt
uv run jupyter notebook submission.ipynb
```

---

## Project Structure

```
analytics_cup_research/
├── submission.ipynb              # Research notebook
├── requirements.txt              # Dependencies
├── docs/
│   └── methodology.md           # Technical derivation of Alvarez target profile
└── src/
    ├── data/loader.py           # Data loading
    ├── analysis/
    │   ├── entries.py           # Entry detection
    │   ├── profiles.py          # Player profiles
    │   ├── similarity.py        # Similarity scoring
    │   └── danger_model.py      # ML model
    └── visualization/
        ├── pitch.py             # Pitch plots
        └── radar.py             # Radar charts
```

For detailed technical documentation on how the Alvarez target profile was derived from StatsBomb data and mapped to SkillCorner metrics, see [docs/methodology.md](docs/methodology.md).

---

SkillCorner X PySport Analytics Cup 2026 — Research Track Submission
