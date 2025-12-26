"""StatsBomb data loading with caching."""

from __future__ import annotations

import polars as pl
from statsbombpy import sb


class StatsBombLoader:
    """Load StatsBomb free data for specific players.

    Uses statsbombpy library to fetch data from StatsBomb's open data repository.
    All data is fetched remotely - no local files required.

    Example:
        loader = StatsBombLoader()
        competitions = loader.get_competitions()
        events = loader.get_player_events("Julián Álvarez", [(43, 106)])
    """

    def __init__(self) -> None:
        """Initialize the loader."""
        self._competitions_cache: pl.DataFrame | None = None
        self._matches_cache: dict[tuple[int, int], pl.DataFrame] = {}

    def get_competitions(self) -> pl.DataFrame:
        """Get all available competitions from StatsBomb free data.

        Returns:
            DataFrame with columns: competition_id, season_id, country_name,
            competition_name, competition_gender, season_name
        """
        if self._competitions_cache is None:
            df = sb.competitions()
            self._competitions_cache = pl.from_pandas(df)
        return self._competitions_cache

    def get_matches(self, competition_id: int, season_id: int) -> pl.DataFrame:
        """Get all matches for a competition/season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            DataFrame with match information including match_id
        """
        cache_key = (competition_id, season_id)
        if cache_key not in self._matches_cache:
            df = sb.matches(competition_id=competition_id, season_id=season_id)
            self._matches_cache[cache_key] = pl.from_pandas(df)
        return self._matches_cache[cache_key]

    def get_events(self, match_id: int) -> pl.DataFrame:
        """Get all events for a single match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            DataFrame with all event data for the match
        """
        df = sb.events(match_id=match_id)
        return pl.from_pandas(df)

    def get_player_events(
        self,
        player_name: str,
        competitions: list[tuple[int, int]],
        verbose: bool = False,
    ) -> pl.DataFrame:
        """Get all events for a player across specified competitions.

        Iterates through all matches in the given competitions, filters events
        by player name, and concatenates results.

        Args:
            player_name: Exact player name as it appears in StatsBomb data
            competitions: List of (competition_id, season_id) tuples
            verbose: If True, print progress

        Returns:
            Concatenated DataFrame of all player events
        """
        all_events: list[pl.DataFrame] = []

        for comp_id, season_id in competitions:
            matches = self.get_matches(comp_id, season_id)
            match_ids = matches["match_id"].to_list()

            if verbose:
                print(f"Loading {len(match_ids)} matches for competition {comp_id}")

            for match_id in match_ids:
                try:
                    events = self.get_events(match_id)

                    # Filter for player events
                    if "player" in events.columns:
                        player_events = events.filter(pl.col("player") == player_name)

                        if len(player_events) > 0:
                            # Add match_id if not present
                            if "match_id" not in player_events.columns:
                                player_events = player_events.with_columns(
                                    pl.lit(match_id).alias("match_id")
                                )
                            all_events.append(player_events)

                            if verbose:
                                print(f"  Match {match_id}: {len(player_events)} events")

                except Exception as e:
                    if verbose:
                        print(f"  Match {match_id}: Error - {e}")
                    continue

        if not all_events:
            return pl.DataFrame()

        # Concatenate with diagonal to handle schema differences
        return pl.concat(all_events, how="diagonal")

    def find_player_in_competition(
        self,
        player_name_partial: str,
        competition_id: int,
        season_id: int,
    ) -> list[str]:
        """Find players matching a partial name in a competition.

        Useful for discovering exact player names in StatsBomb data.

        Args:
            player_name_partial: Partial name to search for (case-insensitive)
            competition_id: Competition to search
            season_id: Season to search

        Returns:
            List of unique player names matching the search
        """
        matches = self.get_matches(competition_id, season_id)

        if len(matches) == 0:
            return []

        # Sample first few matches
        sample_ids = matches["match_id"].head(3).to_list()
        all_players: set[str] = set()

        for match_id in sample_ids:
            try:
                events = self.get_events(match_id)
                if "player" in events.columns:
                    players = events["player"].drop_nulls().unique().to_list()
                    all_players.update(players)
            except Exception:
                continue

        # Filter by partial match
        search_lower = player_name_partial.lower()
        return [p for p in all_players if search_lower in p.lower()]
