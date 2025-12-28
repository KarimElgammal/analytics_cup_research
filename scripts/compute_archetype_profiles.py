"""Compute archetype profiles from StatsBomb open data.

This script fetches event data from StatsBomb's open data repository
and computes defensive/goalkeeper metrics for archetype players.

Source: https://github.com/statsbomb/open-data

Usage:
    uv run python scripts/compute_archetype_profiles.py

Output:
    Prints computed metrics for each player that can be used in app.py
    DEFENDER_ARCHETYPES and GOALKEEPER_ARCHETYPES dictionaries.

To add a new player:
    1. Find their matches in StatsBomb open data (WC22, Euro24, etc.)
    2. Add player name and match IDs to PLAYER_MATCHES dict
    3. Run this script and copy the output to app.py
"""

from statsbombpy import sb
import polars as pl
import warnings
warnings.filterwarnings("ignore")

# Player name -> list of match IDs
PLAYER_MATCHES = {
    # Defenders
    "Joško Gvardiol": [3869519, 3869684, 3857277, 3857290, 3857284],  # WC22 Croatia
    "Virgil van Dijk": [3942819, 3941021, 3942382, 3857295, 3857270, 3857259, 3869118],  # Euro24 + WC22 NED
    "Achraf Hakimi Mouh": [3869552, 3869486, 3869220, 3857283, 3857276, 3857277],  # WC22 Morocco
    # Goalkeepers
    "Manuel Neuer": [3942226, 3938645, 3930158],  # Euro24 Germany
    "Hugo Lloris": [3869354, 3869321, 3869152, 3857299, 3857289, 3857263],  # WC22 France
    "Yassine Bounou": [3869552, 3869486, 3869220, 3857283, 3857276, 3857277],  # WC22 Morocco
}


def get_player_events(player_name: str, match_ids: list[int]) -> pl.DataFrame:
    """Get events for a player from specific matches."""
    all_events = []
    for match_id in match_ids:
        try:
            events_pd = sb.events(match_id=match_id)
            events = pl.from_pandas(events_pd)
            player_events = events.filter(pl.col("player") == player_name)
            if len(player_events) > 0:
                player_events = player_events.with_columns(pl.lit(match_id).alias("match_id"))
                all_events.append(player_events)
                print(f"  {match_id}: {len(player_events)} events")
        except Exception:
            continue
    if not all_events:
        return pl.DataFrame()
    return pl.concat(all_events, how="diagonal")


def compute_defender_metrics(events: pl.DataFrame) -> dict:
    """Compute defender metrics from events."""
    if events.is_empty():
        return {}

    pressures = len(events.filter(pl.col("type") == "Pressure"))
    duels = events.filter(pl.col("type") == "Duel")
    interceptions = len(events.filter(pl.col("type") == "Interception"))
    blocks = len(events.filter(pl.col("type") == "Block"))
    clearances = len(events.filter(pl.col("type") == "Clearance"))
    ball_recoveries = len(events.filter(pl.col("type") == "Ball Recovery"))

    total_defensive = pressures + len(duels) + interceptions + blocks + clearances
    if total_defensive == 0:
        return {"error": "No defensive events"}

    # Pressing rate
    pressing_rate = (pressures / total_defensive * 100)

    # Duel success (stop_danger_rate)
    if "duel_outcome" in duels.columns and len(duels) > 0:
        won = duels.filter(
            pl.col("duel_outcome").is_in(["Won", "Success In Play", "Success Out"])
        )
        stop_danger_rate = (len(won) / len(duels) * 100)
    else:
        stop_danger_rate = 50

    # Goal side rate (blocks + clearances = positional defending)
    goal_side_rate = ((clearances + blocks) / total_defensive * 100)

    # Reduce danger rate (interceptions + recoveries)
    reduce_danger_rate = ((interceptions + ball_recoveries) / (total_defensive + ball_recoveries) * 100)

    # Engagement distance from location (x coordinate)
    engagement_distance = 50
    if "location" in events.columns:
        def_events = events.filter(pl.col("type").is_in(["Pressure", "Duel", "Interception"]))
        locs = def_events.select("location").drop_nulls()
        if len(locs) > 0:
            try:
                # Location is a list [x, y]
                x_coords = [loc[0] for loc in locs["location"].to_list() if loc and len(loc) >= 2]
                if x_coords:
                    avg_x = sum(x_coords) / len(x_coords)
                    engagement_distance = (avg_x / 120) * 100
            except Exception:
                pass

    return {
        "stop_danger_rate": round(stop_danger_rate),
        "reduce_danger_rate": round(reduce_danger_rate),
        "pressing_rate": round(pressing_rate),
        "goal_side_rate": round(min(90, goal_side_rate * 2.5)),
        "beaten_by_movement_rate": round(100 - stop_danger_rate),
        "avg_engagement_distance": round(engagement_distance),
        "total_defensive_events": total_defensive,
        "matches": events["match_id"].n_unique(),
    }


def compute_goalkeeper_metrics(events: pl.DataFrame) -> dict:
    """Compute goalkeeper metrics from events."""
    if events.is_empty():
        return {}

    passes = events.filter(pl.col("type") == "Pass")
    if len(passes) == 0:
        return {"total_events": len(events), "passes": 0}

    # Pass success (null outcome = success)
    if "pass_outcome" in passes.columns:
        successful = len(passes.filter(pl.col("pass_outcome").is_null()))
        pass_success_rate = (successful / len(passes) * 100)
    else:
        pass_success_rate = 70

    # Pass length
    if "pass_length" in passes.columns:
        avg_len = passes["pass_length"].mean() or 30
        pass_distance = min(100, (avg_len / 60) * 100)
        long_passes = len(passes.filter(pl.col("pass_length") > 35))
        long_pass_pct = (long_passes / len(passes) * 100)
    else:
        pass_distance = 50
        long_pass_pct = 30

    return {
        "pass_success_rate": round(pass_success_rate),
        "avg_pass_distance": round(pass_distance),
        "long_pass_pct": round(long_pass_pct),
        "quick_distribution_pct": 50,
        "total_passes": len(passes),
        "matches": events["match_id"].n_unique(),
    }


def main():
    print("=" * 60)
    print("DEFENDER PROFILES FROM STATSBOMB OPEN DATA")
    print("=" * 60)

    for player, match_ids in PLAYER_MATCHES.items():
        if player == "Manuel Neuer":
            continue

        print(f"\n{player}:")
        events = get_player_events(player, match_ids)
        if events.is_empty():
            print("  No events found")
            continue

        metrics = compute_defender_metrics(events)
        print(f"\n  PROFILE (from {metrics.get('matches', 0)} matches, {metrics.get('total_defensive_events', 0)} events):")
        for k, v in metrics.items():
            if k not in ["matches", "total_defensive_events"]:
                print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("GOALKEEPER PROFILES")
    print("=" * 60)

    for player in ["Manuel Neuer", "Hugo Lloris", "Yassine Bounou"]:
        print(f"\n{player}:")
        events = get_player_events(player, PLAYER_MATCHES[player])
        if events.is_empty():
            print("  No events found")
            continue

        metrics = compute_goalkeeper_metrics(events)
        print(f"\n  PROFILE (from {metrics.get('matches', 0)} matches):")
        for k, v in metrics.items():
            if k not in ["matches", "total_passes"]:
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
