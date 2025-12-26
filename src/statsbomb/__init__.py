"""StatsBomb data loading and player statistics module."""

from src.statsbomb.loader import StatsBombLoader
from src.statsbomb.stats import PlayerStats, calculate_player_stats
from src.statsbomb.registry import PLAYER_REGISTRY, get_available_players, get_player_info
from src.statsbomb.mapper import map_to_skillcorner_target

__all__ = [
    "StatsBombLoader",
    "PlayerStats",
    "calculate_player_stats",
    "PLAYER_REGISTRY",
    "get_available_players",
    "get_player_info",
    "map_to_skillcorner_target",
]
