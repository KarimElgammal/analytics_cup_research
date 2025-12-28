"""Archetype definitions and configuration."""

from src.archetypes.base import ArchetypeConfig, Position
from src.archetypes.forwards import FORWARD_WEIGHTS, FORWARD_DIRECTIONS, FORWARD_ARCHETYPE_OPTIONS
from src.archetypes.defenders import DEFENDER_ARCHETYPES, DEFENDER_WEIGHTS, DEFENDER_DIRECTIONS
from src.archetypes.goalkeepers import GOALKEEPER_ARCHETYPES, GOALKEEPER_WEIGHTS, GOALKEEPER_DIRECTIONS

__all__ = [
    "ArchetypeConfig",
    "Position",
    "FORWARD_ARCHETYPE_OPTIONS",
    "FORWARD_WEIGHTS",
    "FORWARD_DIRECTIONS",
    "DEFENDER_ARCHETYPES",
    "DEFENDER_WEIGHTS",
    "DEFENDER_DIRECTIONS",
    "GOALKEEPER_ARCHETYPES",
    "GOALKEEPER_WEIGHTS",
    "GOALKEEPER_DIRECTIONS",
]
