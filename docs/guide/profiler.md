# Building Profiles

The `PlayerProfiler` class handles the pipeline from raw data to player profiles.

## Basic Usage

```python
from src.core import PlayerProfiler

profiler = PlayerProfiler(min_entries=3)
profiler.load_data()
profiler.detect_entries()
profiler.build_profiles()
profiler.add_ages()
```

## One-liner

```python
profiler = PlayerProfiler.from_skillcorner(min_entries=3)
```

## Inspecting Results

```python
summary = profiler.get_summary()
print(f"Profiles: {summary['profiles_built']}")

profile = profiler.get_player_profile("Z. Clough")
print(f"Danger rate: {profile['danger_rate']}%")
```
