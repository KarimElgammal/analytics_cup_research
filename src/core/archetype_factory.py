"""Factory for building player archetypes from StatsBomb data."""

from __future__ import annotations

from src.core.archetype import Archetype
from src.statsbomb.loader import StatsBombLoader
from src.statsbomb.stats import (
    calculate_player_stats,
    calculate_defender_stats,
    calculate_goalkeeper_stats,
)
from src.statsbomb.mappers import ForwardMapper, DefenderMapper, GoalkeeperMapper
from src.statsbomb.registry import (
    get_player_info,
    get_player_competitions,
    get_available_players,
)
from src.archetypes.forwards import FORWARD_DIRECTIONS
from src.archetypes.defenders import DEFENDER_DIRECTIONS
from src.archetypes.goalkeepers import GOALKEEPER_DIRECTIONS


class ArchetypeFactory:
    """Build player archetypes from StatsBomb event data.

    This factory loads real player events from StatsBomb free data,
    calculates statistics, and maps them to SkillCorner-compatible
    target profiles for similarity matching.

    Uses position-specific mappers for clean separation of concerns.

    Example:
        factory = ArchetypeFactory()
        alvarez = factory.build("alvarez")
        print(alvarez.description)
    """

    # Position-specific mappers (class-level for efficiency)
    _forward_mapper = ForwardMapper()
    _defender_mapper = DefenderMapper()
    _goalkeeper_mapper = GoalkeeperMapper()

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the factory.

        Args:
            verbose: If True, print progress when loading data
        """
        self.loader = StatsBombLoader()
        self.verbose = verbose
        self._cache: dict[str, Archetype] = {}

    def build(
        self,
        player_key: str,
        weights: dict[str, float] | None = None,
        directions: dict[str, int] | None = None,
    ) -> Archetype:
        """Build a forward archetype from StatsBomb data.

        Args:
            player_key: Key from PLAYER_REGISTRY (e.g., "alvarez", "messi")
            weights: Optional custom weights (defaults to data-driven)
            directions: Optional custom directions

        Returns:
            Archetype with data-driven target profile and description

        Raises:
            KeyError: If player_key not found in registry
            ValueError: If no events found for the player
        """
        cache_key = player_key
        if cache_key in self._cache and weights is None and directions is None:
            return self._cache[cache_key]

        info = get_player_info(player_key)
        competitions = get_player_competitions(player_key)

        if not competitions:
            raise ValueError(
                f"No competitions defined for player '{player_key}'. "
                "This player may not be available in StatsBomb free data."
            )

        if self.verbose:
            print(f"Loading events for {info['display_name']}...")

        events = self.loader.get_player_events(
            player_name=info["player_name"],
            competitions=competitions,
            verbose=self.verbose,
        )

        if len(events) == 0:
            raise ValueError(
                f"No events found for '{info['player_name']}' in competitions. "
                "Check that the player name matches exactly."
            )

        if self.verbose:
            print(f"Found {len(events)} events, calculating stats...")

        # Calculate statistics and map using ForwardMapper
        stats = calculate_player_stats(events)
        mapper = self._forward_mapper

        target = mapper.map_to_target(stats)
        description = mapper.get_description(stats, target)
        final_weights = weights if weights is not None else mapper.compute_weights(stats)
        final_directions = directions if directions is not None else FORWARD_DIRECTIONS.copy()

        archetype = Archetype(
            name=player_key,
            description=f"{info['display_name']}\n{description}",
            target_profile=target,
            weights=final_weights,
            directions=final_directions,
        )

        if weights is None and directions is None:
            self._cache[cache_key] = archetype

        return archetype

    def build_defender(self, player_key: str) -> Archetype:
        """Build a defender archetype from StatsBomb data.

        Args:
            player_key: Key from PLAYER_REGISTRY (e.g., "gvardiol", "vandijk")

        Returns:
            Archetype with data-driven target profile
        """
        cache_key = f"defender_{player_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        info = get_player_info(player_key)
        competitions = get_player_competitions(player_key)

        if not competitions:
            raise ValueError(f"No competitions for defender '{player_key}'")

        if self.verbose:
            print(f"Loading events for {info['display_name']}...")

        events = self.loader.get_player_events(
            player_name=info["player_name"],
            competitions=competitions,
            verbose=self.verbose,
        )

        if len(events) == 0:
            raise ValueError(f"No events found for '{info['player_name']}'")

        if self.verbose:
            print(f"Found {len(events)} events, calculating defender stats...")

        # Calculate statistics and map using DefenderMapper
        stats = calculate_defender_stats(events)
        mapper = self._defender_mapper

        target = mapper.map_to_target(stats)
        description = mapper.get_description(stats, target)
        weights = mapper.compute_weights(stats)

        archetype = Archetype(
            name=player_key,
            description=f"{info['display_name']}\n{description}",
            target_profile=target,
            weights=weights,
            directions=DEFENDER_DIRECTIONS.copy(),
        )

        self._cache[cache_key] = archetype
        return archetype

    def build_goalkeeper(self, player_key: str) -> Archetype:
        """Build a goalkeeper archetype from StatsBomb data.

        Args:
            player_key: Key from PLAYER_REGISTRY (e.g., "neuer", "lloris")

        Returns:
            Archetype with data-driven target profile
        """
        cache_key = f"goalkeeper_{player_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        info = get_player_info(player_key)
        competitions = get_player_competitions(player_key)

        if not competitions:
            raise ValueError(f"No competitions for goalkeeper '{player_key}'")

        if self.verbose:
            print(f"Loading events for {info['display_name']}...")

        events = self.loader.get_player_events(
            player_name=info["player_name"],
            competitions=competitions,
            verbose=self.verbose,
        )

        if len(events) == 0:
            raise ValueError(f"No events found for '{info['player_name']}'")

        if self.verbose:
            print(f"Found {len(events)} events, calculating goalkeeper stats...")

        # Calculate statistics and map using GoalkeeperMapper
        stats = calculate_goalkeeper_stats(events)
        mapper = self._goalkeeper_mapper

        target = mapper.map_to_target(stats)
        description = mapper.get_description(stats, target)
        weights = mapper.compute_weights(stats)

        archetype = Archetype(
            name=player_key,
            description=f"{info['display_name']}\n{description}",
            target_profile=target,
            weights=weights,
            directions=GOALKEEPER_DIRECTIONS.copy(),
        )

        self._cache[cache_key] = archetype
        return archetype

    def list_available(self) -> list[str]:
        """Get list of available player archetypes."""
        return get_available_players()

    def get_dropdown_options(self) -> list[tuple[str, str]]:
        """Get options formatted for dropdown widget."""
        options = []
        for key in self.list_available():
            try:
                info = get_player_info(key)
                comps = info.get("competitions", [])
                if comps:
                    comp_name = comps[0][2] if len(comps[0]) > 2 else "Unknown"
                    display = f"{info['display_name']} ({comp_name})"
                else:
                    display = info["display_name"]
                options.append((display, key))
            except KeyError:
                continue
        return options

    def clear_cache(self) -> None:
        """Clear the archetype cache."""
        self._cache.clear()


# Module-level convenience functions
_default_factory: ArchetypeFactory | None = None


def get_factory() -> ArchetypeFactory:
    """Get or create the default factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = ArchetypeFactory()
    return _default_factory


def build_archetype(player_key: str) -> Archetype:
    """Convenience function to build an archetype."""
    return get_factory().build(player_key)


def list_archetypes() -> list[str]:
    """Convenience function to list available archetypes."""
    return get_factory().list_available()
