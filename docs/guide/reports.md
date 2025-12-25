# AI Reports

The `ScoutingReport` class generates AI-powered scouting insights.

## Setup

```python
from src.core import ScoutingReport

report = ScoutingReport(engine)
report.set_token("your_github_token")
```

## Generate Report

```python
if report.has_valid_token():
    insight = report.generate(top_n=5)
    print(insight)
```

## Individual Player Report

```python
player_insight = report.player_report("Z. Clough")
print(player_insight)
```
