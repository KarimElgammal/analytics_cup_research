"""Factory for building player archetypes from StatsBomb data."""

from __future__ import annotations

from src.core.archetype import Archetype
from src.statsbomb.loader import StatsBombLoader
from src.statsbomb.stats import calculate_player_stats
from src.statsbomb.mapper import map_to_skillcorner_target, get_archetype_description
from src.statsbomb.registry import get_player_info, get_player_competitions, get_available_players


# ML-calibrated weights from existing danger model analysis
# Cross-validated AUC: 0.656 on 245 A-League entries
# These weights determine feature importance in similarity scoring
DEFAULT_WEIGHTS = {
    "avg_separation": 0.23,           # Highest ML importance - finding space
    "danger_rate": 0.18,              # Clinical finishing
    "avg_entry_speed": 0.17,          # Dynamic entries
    "avg_defensive_line_dist": 0.15,  # Penetration depth
    "central_pct": 0.12,              # Central positioning
    "quick_break_pct": 0.05,          # Counter-attack involvement
    "avg_teammates_ahead": 0.05,      # Link-up context
    "half_space_pct": 0.02,           # Half-space usage
    "avg_passing_options": 0.02,      # Passing options
    "carry_pct": 0.00,                # ML confirmed low importance
    "avg_distance": 0.01,             # Work rate
    "goal_rate": 0.00,                # Too sparse in data
}

# Feature directions: 1 = higher is better, -1 = lower is better
DEFAULT_DIRECTIONS = {
    "avg_separation": 1,
    "danger_rate": 1,
    "avg_entry_speed": 1,
    "avg_defensive_line_dist": -1,  # Closer to goal is better
    "central_pct": 1,
    "quick_break_pct": 1,
    "avg_teammates_ahead": 1,
    "half_space_pct": 1,
    "avg_passing_options": 1,
    "carry_pct": 1,
    "avg_distance": 1,
    "goal_rate": 1,
}


class ArchetypeFactory:
    """Build player archetypes from StatsBomb event data.

    This factory loads real player events from StatsBomb free data,
    calculates statistics, and maps them to SkillCorner-compatible
    target profiles for similarity matching.

    Example:
        factory = ArchetypeFactory()
        alvarez = factory.build("alvarez")
        print(alvarez.description)
        # Shows actual stats from World Cup 2022

    The factory caches built archetypes to avoid redundant API calls.
    """

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
        """Build an archetype from StatsBomb data for a registered player.

        Args:
            player_key: Key from PLAYER_REGISTRY (e.g., "alvarez", "messi")
            weights: Optional custom weights (defaults to ML-calibrated)
            directions: Optional custom directions (defaults to standard)

        Returns:
            Archetype with data-driven target profile and description

        Raises:
            KeyError: If player_key not found in registry
            ValueError: If no events found for the player
        """
        # Return cached if available
        if player_key in self._cache and weights is None and directions is None:
            return self._cache[player_key]

        # Get player info from registry
        info = get_player_info(player_key)
        competitions = get_player_competitions(player_key)

        if not competitions:
            raise ValueError(
                f"No competitions defined for player '{player_key}'. "
                "This player may not be available in StatsBomb free data."
            )

        if self.verbose:
            print(f"Loading events for {info['display_name']}...")

        # Load player events from StatsBomb
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

        # Calculate statistics
        stats = calculate_player_stats(events)

        # Map to SkillCorner target profile
        target = map_to_skillcorner_target(stats)

        # Generate description
        description = get_archetype_description(stats, target)

        # Use provided weights/directions or defaults
        final_weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        final_directions = directions if directions is not None else DEFAULT_DIRECTIONS.copy()

        # Build the archetype
        archetype = Archetype(
            name=player_key,
            description=f"{info['display_name']}\n{description}",
            target_profile=target,
            weights=final_weights,
            directions=final_directions,
        )

        # Cache if using defaults
        if weights is None and directions is None:
            self._cache[player_key] = archetype

        return archetype

    def list_available(self) -> list[str]:
        """Get list of available player archetypes.

        Returns:
            List of player keys that can be used with build()
        """
        return get_available_players()

    def get_dropdown_options(self) -> list[tuple[str, str]]:
        """Get options formatted for notebook dropdown widget.

        Returns:
            List of (display_name, key) tuples for ipywidgets.Dropdown
        """
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
    """Convenience function to build an archetype.

    Args:
        player_key: Key from PLAYER_REGISTRY

    Returns:
        Archetype built from StatsBomb data
    """
    return get_factory().build(player_key)


def list_archetypes() -> list[str]:
    """Convenience function to list available archetypes.

    Returns:
        List of available player keys
    """
    return get_factory().list_available()
