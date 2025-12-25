# Getting Started

## Installation

```bash
git clone https://github.com/KarimElgammal/analytics_cup_research.git
cd analytics_cup_research
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Basic Usage

```python
from src.core import Archetype, PlayerProfiler, SimilarityEngine

profiler = PlayerProfiler(min_entries=3)
profiler.load_data()
profiler.detect_entries()
profiler.build_profiles()

archetype = Archetype.alvarez()
engine = SimilarityEngine(archetype)
engine.fit(profiler.profiles)
rankings = engine.rank(top_n=10)
```
