"""Player statistics calculation from StatsBomb event data."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class PlayerStats:
    """Computed statistics from StatsBomb events.

    All percentage values are stored as 0-100 scale.

    Attributes:
        player_name: Name of the player
        matches: Number of matches with events
        minutes: Estimated minutes played

        shots: Total shots taken
        shots_on_target: Shots that hit the target (saved or goal)
        goals: Goals scored
        shot_accuracy: Percentage of shots on target
        conversion_rate: Percentage of shots that became goals

        passes: Total passes attempted
        passes_completed: Successful passes
        pass_accuracy: Percentage of completed passes
        key_passes: Passes that led to shots (shot assists)

        dribbles: Total dribble attempts
        dribbles_completed: Successful dribbles
        dribble_success: Percentage of successful dribbles

        box_touches: Events in the penalty box (x >= 102)
    """

    player_name: str
    matches: int
    minutes: float

    # Shooting
    shots: int
    shots_on_target: int
    goals: int
    shot_accuracy: float
    conversion_rate: float

    # Passing
    passes: int
    passes_completed: int
    pass_accuracy: float
    key_passes: int

    # Dribbling
    dribbles: int
    dribbles_completed: int
    dribble_success: float

    # Box presence
    box_touches: int

    @property
    def goals_per_90(self) -> float:
        """Goals per 90 minutes played."""
        return (self.goals / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def shots_per_90(self) -> float:
        """Shots per 90 minutes played."""
        return (self.shots / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def box_touches_per_90(self) -> float:
        """Box touches per 90 minutes played."""
        return (self.box_touches / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def key_passes_per_90(self) -> float:
        """Key passes per 90 minutes played."""
        return (self.key_passes / self.minutes) * 90 if self.minutes > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for display/export."""
        return {
            "player_name": self.player_name,
            "matches": self.matches,
            "minutes": self.minutes,
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "goals": self.goals,
            "shot_accuracy": round(self.shot_accuracy, 1),
            "conversion_rate": round(self.conversion_rate, 1),
            "passes": self.passes,
            "passes_completed": self.passes_completed,
            "pass_accuracy": round(self.pass_accuracy, 1),
            "key_passes": self.key_passes,
            "dribbles": self.dribbles,
            "dribbles_completed": self.dribbles_completed,
            "dribble_success": round(self.dribble_success, 1),
            "box_touches": self.box_touches,
            "goals_per_90": round(self.goals_per_90, 2),
            "box_touches_per_90": round(self.box_touches_per_90, 2),
        }


def calculate_player_stats(events: pl.DataFrame) -> PlayerStats:
    """Calculate statistics from a DataFrame of player events.

    Expects events already filtered to a single player.
    Handles StatsBomb event data format with type, shot_outcome, etc.

    Args:
        events: DataFrame of StatsBomb events for one player

    Returns:
        PlayerStats dataclass with computed statistics
    """
    if len(events) == 0:
        return PlayerStats(
            player_name="Unknown",
            matches=0,
            minutes=0,
            shots=0,
            shots_on_target=0,
            goals=0,
            shot_accuracy=0,
            conversion_rate=0,
            passes=0,
            passes_completed=0,
            pass_accuracy=0,
            key_passes=0,
            dribbles=0,
            dribbles_completed=0,
            dribble_success=0,
            box_touches=0,
        )

    # Get player name from first event
    player_name = ""
    if "player" in events.columns:
        first_player = events["player"].drop_nulls().first()
        player_name = first_player if first_player else "Unknown"

    # Count matches
    matches = 0
    if "match_id" in events.columns:
        matches = events.select(pl.col("match_id").n_unique()).item()

    # Estimate minutes from event timestamps
    # Use unique minute values as a rough approximation
    minutes = 1.0
    if "minute" in events.columns:
        unique_minutes = events.select(pl.col("minute").n_unique()).item()
        minutes = max(unique_minutes or 1, 1)

    # === SHOTS ===
    shots = 0
    shots_on_target = 0
    goals = 0

    if "type" in events.columns:
        shots_df = events.filter(pl.col("type") == "Shot")
        shots = len(shots_df)

        if "shot_outcome" in events.columns and shots > 0:
            goals = len(shots_df.filter(pl.col("shot_outcome") == "Goal"))
            shots_on_target = len(
                shots_df.filter(pl.col("shot_outcome").is_in(["Goal", "Saved"]))
            )

    shot_accuracy = (shots_on_target / shots * 100) if shots > 0 else 0.0
    conversion_rate = (goals / shots * 100) if shots > 0 else 0.0

    # === PASSES ===
    passes = 0
    passes_completed = 0
    key_passes = 0

    if "type" in events.columns:
        passes_df = events.filter(pl.col("type") == "Pass")
        passes = len(passes_df)

        if passes > 0:
            # Successful passes have null outcome (no failure recorded)
            if "pass_outcome" in events.columns:
                passes_completed = len(passes_df.filter(pl.col("pass_outcome").is_null()))
            else:
                passes_completed = passes  # Assume all complete if no outcome column

            # Key passes = shot assists
            if "pass_shot_assist" in events.columns:
                key_passes = len(
                    passes_df.filter(pl.col("pass_shot_assist") == True)
                )

    pass_accuracy = (passes_completed / passes * 100) if passes > 0 else 0.0

    # === DRIBBLES ===
    dribbles = 0
    dribbles_completed = 0

    if "type" in events.columns:
        dribbles_df = events.filter(pl.col("type") == "Dribble")
        dribbles = len(dribbles_df)

        if dribbles > 0 and "dribble_outcome" in events.columns:
            dribbles_completed = len(
                dribbles_df.filter(pl.col("dribble_outcome") == "Complete")
            )

    dribble_success = (dribbles_completed / dribbles * 100) if dribbles > 0 else 0.0

    # === BOX TOUCHES ===
    # In StatsBomb coordinates, x >= 102 is in the penalty area
    box_touches = 0

    if "location" in events.columns:
        try:
            # location is typically a list [x, y]
            box_touches = len(
                events.filter(pl.col("location").list.get(0) >= 102)
            )
        except Exception:
            # Handle case where location format differs
            box_touches = 0

    return PlayerStats(
        player_name=player_name,
        matches=matches,
        minutes=minutes,
        shots=shots,
        shots_on_target=shots_on_target,
        goals=goals,
        shot_accuracy=shot_accuracy,
        conversion_rate=conversion_rate,
        passes=passes,
        passes_completed=passes_completed,
        pass_accuracy=pass_accuracy,
        key_passes=key_passes,
        dribbles=dribbles,
        dribbles_completed=dribbles_completed,
        dribble_success=dribble_success,
        box_touches=box_touches,
    )
