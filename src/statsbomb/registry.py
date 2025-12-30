"""Registry of known players available in StatsBomb free data."""

from __future__ import annotations

# Known players in StatsBomb FREE data (World Cup, Copa America, etc.)
# Player names must match exactly as they appear in StatsBomb event data
#
# Selection criteria for archetypes:
# - Players with sufficient data points (5+ shots in tournament)
# - Represent different playing styles for meaningful A-League comparisons
# - More "realistic" targets than generational talents like Messi/Mbappe
PLAYER_REGISTRY: dict[str, dict] = {
    "alvarez": {
        "player_name": "Julián Álvarez",
        "display_name": "Julian Alvarez",
        "position": "Forward",
        "nationality": "Argentina",
        "style": "Movement-focused, intelligent runs, clinical finishing",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "giroud": {
        "player_name": "Olivier Giroud",
        "display_name": "Olivier Giroud",
        "position": "Forward",
        "nationality": "France",
        "style": "Target man, hold-up play, aerial threat, 0% dribble reliance",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "kane": {
        "player_name": "Harry Kane",
        "display_name": "Harry Kane",
        "position": "Forward",
        "nationality": "England",
        "style": "Complete forward, link-up play, clinical finishing",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "lewandowski": {
        "player_name": "Robert Lewandowski",
        "display_name": "Robert Lewandowski",
        "position": "Forward",
        "nationality": "Poland",
        "style": "Clinical poacher, box presence, intelligent movement",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "rashford": {
        "player_name": "Marcus Rashford",
        "display_name": "Marcus Rashford",
        "position": "Forward",
        "nationality": "England",
        "style": "Pace, direct dribbling, wide threat",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "en_nesyri": {
        "player_name": "Youssef En-Nesyri",
        "display_name": "Youssef En-Nesyri",
        "position": "Forward",
        "nationality": "Morocco",
        "style": "Physical forward, aerial presence, direct runs",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    # === DEFENDERS ===
    "gvardiol": {
        "player_name": "Joško Gvardiol",
        "display_name": "Josko Gvardiol",
        "position": "Defender",
        "nationality": "Croatia",
        "style": "Ball-playing CB, progressive carrier, aerial dominant",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "vandijk": {
        "player_name": "Virgil van Dijk",
        "display_name": "Virgil van Dijk",
        "position": "Defender",
        "nationality": "Netherlands",
        "style": "Commanding CB, excellent positioning, aerial dominant",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "hakimi": {
        "player_name": "Achraf Hakimi Mouh",
        "display_name": "Achraf Hakimi",
        "position": "Defender",
        "nationality": "Morocco",
        "style": "Attacking wing-back, pace, overlapping runs, crossing ability",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    # === GOALKEEPERS ===
    "lloris": {
        "player_name": "Hugo Lloris",
        "display_name": "Hugo Lloris",
        "position": "Goalkeeper",
        "nationality": "France",
        "style": "Experienced captain, shot-stopper, commanding presence",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "neuer": {
        "player_name": "Manuel Neuer",
        "display_name": "Manuel Neuer",
        "position": "Goalkeeper",
        "nationality": "Germany",
        "style": "Modern sweeper-keeper, excellent distribution, commanding",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "bounou": {
        "player_name": "Yassine Bounou",
        "display_name": "Yassine Bounou",
        "position": "Goalkeeper",
        "nationality": "Morocco",
        "style": "Long distribution, direct style, penalty specialist",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    # === MIDFIELDERS ===
    "enzo": {
        "player_name": "Enzo Fernandez",
        "display_name": "Enzo Fernandez",
        "position": "Midfielder",
        "nationality": "Argentina",
        "style": "Box-to-box, progressive passing, ball winner",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "tchouameni": {
        "player_name": "Aurélien Djani Tchouaméni",
        "display_name": "Aurelien Tchouameni",
        "position": "Midfielder",
        "nationality": "France",
        "style": "Defensive anchor, excellent positioning, high pressing",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "depaul": {
        "player_name": "Rodrigo Javier De Paul",
        "display_name": "Rodrigo De Paul",
        "position": "Midfielder",
        "nationality": "Argentina",
        "style": "Engine, high work rate, link-up play",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "griezmann": {
        "player_name": "Antoine Griezmann",
        "display_name": "Antoine Griezmann",
        "position": "Midfielder",
        "nationality": "France",
        "style": "Attacking midfielder, creativity, chance creation",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "pedri": {
        "player_name": "Pedro González López",
        "display_name": "Pedri",
        "position": "Midfielder",
        "nationality": "Spain",
        "style": "Technical control, possession retention, low turnovers",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
    "bellingham": {
        "player_name": "Jude Bellingham",
        "display_name": "Jude Bellingham",
        "position": "Midfielder",
        "nationality": "England",
        "style": "Complete midfielder, goal threat, box arrivals",
        "competitions": [
            (43, 106, "FIFA World Cup 2022"),
        ],
    },
}


def get_available_players() -> list[str]:
    """Return list of player keys with available competition data."""
    return [
        key for key, info in PLAYER_REGISTRY.items()
        if info.get("competitions")  # Only return players with actual competition data
    ]


def get_player_info(player_key: str) -> dict:
    """Get player metadata including name and competitions.

    Args:
        player_key: Key from PLAYER_REGISTRY (e.g., "alvarez", "messi")

    Returns:
        Dictionary with player_name, display_name, position, competitions

    Raises:
        KeyError: If player_key not found in registry
    """
    if player_key not in PLAYER_REGISTRY:
        available = ", ".join(get_available_players())
        raise KeyError(f"Player '{player_key}' not found. Available: {available}")
    return PLAYER_REGISTRY[player_key]


def get_player_competitions(player_key: str) -> list[tuple[int, int]]:
    """Get (competition_id, season_id) tuples for a player.

    Args:
        player_key: Key from PLAYER_REGISTRY

    Returns:
        List of (competition_id, season_id) tuples
    """
    info = get_player_info(player_key)
    return [(c[0], c[1]) for c in info.get("competitions", [])]
