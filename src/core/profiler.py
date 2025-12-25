"""Player profile building from SkillCorner tracking data."""

from __future__ import annotations
import polars as pl


class PlayerProfiler:
    """Build player profiles from SkillCorner tracking data."""

    def __init__(self, min_entries: int = 3):
        self.min_entries = min_entries
        self.events: pl.DataFrame | None = None
        self.entries: pl.DataFrame | None = None
        self.profiles: pl.DataFrame | None = None

    def load_data(self, match_ids: list[int] | None = None) -> PlayerProfiler:
        from src.data.loader import load_all_events, add_team_names
        self.events = load_all_events(match_ids=match_ids)
        self.events = add_team_names(self.events)
        return self

    def load_from_dataframe(self, events: pl.DataFrame) -> PlayerProfiler:
        self.events = events
        return self

    def detect_entries(self) -> PlayerProfiler:
        if self.events is None:
            raise ValueError("No events loaded. Call load_data() first.")
        from src.analysis.entries import detect_entries, classify_entries
        self.entries = detect_entries(self.events)
        self.entries = classify_entries(self.entries)
        return self

    def build_profiles(self) -> PlayerProfiler:
        if self.entries is None:
            raise ValueError("No entries detected. Call detect_entries() first.")
        from src.analysis.profiles import build_player_profiles, filter_profiles
        self.profiles = build_player_profiles(self.entries)
        self.profiles = filter_profiles(self.profiles, min_entries=self.min_entries)
        return self

    def add_ages(self, ages: dict[str, int] | None = None) -> PlayerProfiler:
        if self.profiles is None:
            raise ValueError("No profiles built. Call build_profiles() first.")
        if ages is None:
            from src.data.player_ages import add_ages_to_profiles
            self.profiles = add_ages_to_profiles(self.profiles)
        else:
            age_data = [(name, age) for name, age in ages.items()]
            age_df = pl.DataFrame(age_data, schema=["player_name", "age"], orient="row")
            self.profiles = self.profiles.join(age_df, on="player_name", how="left")
        return self

    def get_summary(self) -> dict:
        return {
            "events_loaded": len(self.events) if self.events is not None else 0,
            "entries_detected": len(self.entries) if self.entries is not None else 0,
            "profiles_built": len(self.profiles) if self.profiles is not None else 0,
        }

    def get_player_entries(self, player_name: str) -> pl.DataFrame | None:
        if self.entries is None:
            return None
        result = self.entries.filter(pl.col("player_name") == player_name)
        return result if len(result) > 0 else None

    def get_player_profile(self, player_name: str) -> dict | None:
        if self.profiles is None:
            return None
        result = self.profiles.filter(pl.col("player_name") == player_name)
        return result.to_dicts()[0] if len(result) > 0 else None

    @classmethod
    def from_skillcorner(cls, match_ids: list[int] | None = None, min_entries: int = 3) -> PlayerProfiler:
        profiler = cls(min_entries=min_entries)
        profiler.load_data(match_ids=match_ids)
        profiler.detect_entries()
        profiler.build_profiles()
        profiler.add_ages()
        return profiler
