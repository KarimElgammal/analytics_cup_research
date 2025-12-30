"""Player statistics calculation from StatsBomb event data."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class DefenderStats:
    """Computed statistics for defenders from StatsBomb events.

    Attributes:
        player_name: Name of the player
        matches: Number of matches with events
        minutes: Estimated minutes played

        tackles: Total tackle attempts
        tackles_won: Successful tackles
        interceptions: Ball interceptions
        clearances: Ball clearances
        blocks: Shot/pass blocks

        pressures: Total pressure events
        pressure_success: Successful pressures (regained possession)

        duels: Total duels
        duels_won: Duels won
        aerial_duels: Total aerial duels
        aerial_duels_won: Aerial duels won

        fouls_committed: Fouls committed
        fouls_won: Fouls won (drawn)

        carries: Ball carries
        progressive_carries: Carries that advanced play significantly
    """

    player_name: str
    matches: int
    minutes: float

    # Defensive actions
    tackles: int
    tackles_won: int
    interceptions: int
    clearances: int
    blocks: int

    # Pressing
    pressures: int
    pressure_success: int

    # Duels
    duels: int
    duels_won: int
    aerial_duels: int
    aerial_duels_won: int

    # Fouls
    fouls_committed: int
    fouls_won: int

    # Ball progression
    carries: int
    progressive_carries: int

    @property
    def tackle_success_rate(self) -> float:
        """Percentage of successful tackles."""
        return (self.tackles_won / self.tackles * 100) if self.tackles > 0 else 0.0

    @property
    def duel_success_rate(self) -> float:
        """Percentage of duels won."""
        return (self.duels_won / self.duels * 100) if self.duels > 0 else 0.0

    @property
    def aerial_success_rate(self) -> float:
        """Percentage of aerial duels won."""
        return (self.aerial_duels_won / self.aerial_duels * 100) if self.aerial_duels > 0 else 0.0

    @property
    def pressure_success_rate(self) -> float:
        """Percentage of successful pressures."""
        return (self.pressure_success / self.pressures * 100) if self.pressures > 0 else 0.0

    @property
    def progressive_carry_pct(self) -> float:
        """Percentage of carries that were progressive."""
        return (self.progressive_carries / self.carries * 100) if self.carries > 0 else 0.0

    @property
    def pressures_per_90(self) -> float:
        """Pressures per 90 minutes."""
        return (self.pressures / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def tackles_per_90(self) -> float:
        """Tackles per 90 minutes."""
        return (self.tackles / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def interceptions_per_90(self) -> float:
        """Interceptions per 90 minutes."""
        return (self.interceptions / self.minutes) * 90 if self.minutes > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for display/export."""
        return {
            "player_name": self.player_name,
            "matches": self.matches,
            "minutes": self.minutes,
            "tackles": self.tackles,
            "tackles_won": self.tackles_won,
            "tackle_success_rate": round(self.tackle_success_rate, 1),
            "interceptions": self.interceptions,
            "clearances": self.clearances,
            "blocks": self.blocks,
            "pressures": self.pressures,
            "pressure_success_rate": round(self.pressure_success_rate, 1),
            "duels": self.duels,
            "duels_won": self.duels_won,
            "duel_success_rate": round(self.duel_success_rate, 1),
            "aerial_duels": self.aerial_duels,
            "aerial_success_rate": round(self.aerial_success_rate, 1),
            "fouls_committed": self.fouls_committed,
            "fouls_won": self.fouls_won,
            "carries": self.carries,
            "progressive_carry_pct": round(self.progressive_carry_pct, 1),
        }


@dataclass
class GoalkeeperStats:
    """Computed statistics for goalkeepers from StatsBomb events.

    Attributes:
        player_name: Name of the player
        matches: Number of matches with events
        minutes: Estimated minutes played

        saves: Total saves
        goals_conceded: Goals conceded

        passes: Total passes/distributions
        passes_completed: Successful passes
        long_passes: Long passes (>32m)
        short_passes: Short passes (<20m)
        high_passes: High/lofted passes

        avg_pass_distance: Average pass distance in meters
    """

    player_name: str
    matches: int
    minutes: float

    # Shot stopping
    saves: int
    goals_conceded: int

    # Distribution
    passes: int
    passes_completed: int
    long_passes: int
    short_passes: int
    high_passes: int

    # Pass metrics
    total_pass_distance: float  # Sum of all pass distances

    @property
    def save_percentage(self) -> float:
        """Percentage of shots saved."""
        total = self.saves + self.goals_conceded
        return (self.saves / total * 100) if total > 0 else 0.0

    @property
    def pass_success_rate(self) -> float:
        """Percentage of successful passes."""
        return (self.passes_completed / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def long_pass_pct(self) -> float:
        """Percentage of passes that were long."""
        return (self.long_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def short_pass_pct(self) -> float:
        """Percentage of passes that were short."""
        return (self.short_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def high_pass_pct(self) -> float:
        """Percentage of passes that were high/lofted."""
        return (self.high_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def avg_pass_distance(self) -> float:
        """Average pass distance in meters."""
        return (self.total_pass_distance / self.passes) if self.passes > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for display/export."""
        return {
            "player_name": self.player_name,
            "matches": self.matches,
            "minutes": self.minutes,
            "saves": self.saves,
            "goals_conceded": self.goals_conceded,
            "save_percentage": round(self.save_percentage, 1),
            "passes": self.passes,
            "passes_completed": self.passes_completed,
            "pass_success_rate": round(self.pass_success_rate, 1),
            "long_passes": self.long_passes,
            "long_pass_pct": round(self.long_pass_pct, 1),
            "short_passes": self.short_passes,
            "short_pass_pct": round(self.short_pass_pct, 1),
            "high_passes": self.high_passes,
            "high_pass_pct": round(self.high_pass_pct, 1),
            "avg_pass_distance": round(self.avg_pass_distance, 1),
        }


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


def calculate_defender_stats(events: pl.DataFrame) -> DefenderStats:
    """Calculate defender statistics from StatsBomb events.

    Args:
        events: DataFrame of StatsBomb events for one defender

    Returns:
        DefenderStats dataclass with computed statistics
    """
    if len(events) == 0:
        return DefenderStats(
            player_name="Unknown", matches=0, minutes=0,
            tackles=0, tackles_won=0, interceptions=0, clearances=0, blocks=0,
            pressures=0, pressure_success=0, duels=0, duels_won=0,
            aerial_duels=0, aerial_duels_won=0, fouls_committed=0, fouls_won=0,
            carries=0, progressive_carries=0,
        )

    # Get player name
    player_name = ""
    if "player" in events.columns:
        first_player = events["player"].drop_nulls().first()
        player_name = first_player if first_player else "Unknown"

    # Count matches
    matches = 0
    if "match_id" in events.columns:
        matches = events.select(pl.col("match_id").n_unique()).item()

    # Estimate minutes
    minutes = 1.0
    if "minute" in events.columns:
        unique_minutes = events.select(pl.col("minute").n_unique()).item()
        minutes = max(unique_minutes or 1, 1)

    # === TACKLES ===
    tackles = 0
    tackles_won = 0
    if "type" in events.columns:
        tackles_df = events.filter(pl.col("type") == "Duel")
        if "duel_type" in events.columns:
            tackles_df = tackles_df.filter(pl.col("duel_type") == "Tackle")
        tackles = len(tackles_df)
        if "duel_outcome" in events.columns and tackles > 0:
            tackles_won = len(tackles_df.filter(
                pl.col("duel_outcome").is_in(["Won", "Success", "Success In Play"])
            ))

    # === INTERCEPTIONS ===
    interceptions = 0
    if "type" in events.columns:
        interceptions = len(events.filter(pl.col("type") == "Interception"))

    # === CLEARANCES ===
    clearances = 0
    if "type" in events.columns:
        clearances = len(events.filter(pl.col("type") == "Clearance"))

    # === BLOCKS ===
    blocks = 0
    if "type" in events.columns:
        blocks = len(events.filter(pl.col("type") == "Block"))

    # === PRESSURES ===
    pressures = 0
    pressure_success = 0
    if "type" in events.columns:
        pressure_df = events.filter(pl.col("type") == "Pressure")
        pressures = len(pressure_df)
        if "counterpress" in events.columns and pressures > 0:
            pressure_success = len(pressure_df.filter(pl.col("counterpress") == True))

    # === DUELS (all types) ===
    duels = 0
    duels_won = 0
    if "type" in events.columns:
        duels_df = events.filter(pl.col("type") == "Duel")
        duels = len(duels_df)
        if "duel_outcome" in events.columns and duels > 0:
            duels_won = len(duels_df.filter(
                pl.col("duel_outcome").is_in(["Won", "Success", "Success In Play"])
            ))

    # === AERIAL DUELS ===
    aerial_duels = 0
    aerial_duels_won = 0
    if "type" in events.columns and "duel_type" in events.columns:
        aerial_df = events.filter(
            (pl.col("type") == "Duel") & (pl.col("duel_type") == "Aerial Lost")
            | (pl.col("type") == "Duel") & (pl.col("duel_type").str.contains("Aerial"))
        )
        aerial_duels = len(aerial_df)
        if "duel_outcome" in events.columns and aerial_duels > 0:
            aerial_duels_won = len(aerial_df.filter(
                pl.col("duel_outcome").is_in(["Won", "Success"])
            ))

    # === FOULS ===
    fouls_committed = 0
    fouls_won = 0
    if "type" in events.columns:
        fouls_committed = len(events.filter(pl.col("type") == "Foul Committed"))
        fouls_won = len(events.filter(pl.col("type") == "Foul Won"))

    # === CARRIES ===
    carries = 0
    progressive_carries = 0
    if "type" in events.columns:
        carry_df = events.filter(pl.col("type") == "Carry")
        carries = len(carry_df)
        # Progressive carry: moved ball forward significantly (use end_location if available)
        if "carry_end_location" in events.columns and carries > 0:
            # Simplified: count carries that end in attacking half (x > 60)
            try:
                progressive_carries = len(carry_df.filter(
                    pl.col("carry_end_location").list.get(0) > 60
                ))
            except Exception:
                progressive_carries = 0

    return DefenderStats(
        player_name=player_name,
        matches=matches,
        minutes=minutes,
        tackles=tackles,
        tackles_won=tackles_won,
        interceptions=interceptions,
        clearances=clearances,
        blocks=blocks,
        pressures=pressures,
        pressure_success=pressure_success,
        duels=duels,
        duels_won=duels_won,
        aerial_duels=aerial_duels,
        aerial_duels_won=aerial_duels_won,
        fouls_committed=fouls_committed,
        fouls_won=fouls_won,
        carries=carries,
        progressive_carries=progressive_carries,
    )


@dataclass
class MidfielderStats:
    """Computed statistics for midfielders from StatsBomb events.

    Attributes:
        player_name: Name of the player
        matches: Number of matches with events
        minutes: Estimated minutes played

        passes: Total passes attempted
        passes_completed: Successful passes
        progressive_passes: Passes that advance play significantly
        key_passes: Passes leading to shots

        carries: Ball carries
        progressive_carries: Carries that advance play significantly

        pressures: Pressure events
        pressure_success: Successful pressures (regained possession)

        tackles: Tackle attempts
        tackles_won: Successful tackles
        interceptions: Ball interceptions
        ball_recoveries: Ball recovery events

        through_balls: Through ball passes
        final_third_passes: Passes into final third
    """

    player_name: str
    matches: int
    minutes: float

    # Passing
    passes: int
    passes_completed: int
    progressive_passes: int
    key_passes: int

    # Ball progression
    carries: int
    progressive_carries: int

    # Defensive contribution
    pressures: int
    pressure_success: int
    tackles: int
    tackles_won: int
    interceptions: int
    ball_recoveries: int

    # Creativity
    through_balls: int
    final_third_passes: int

    @property
    def pass_accuracy(self) -> float:
        """Percentage of completed passes."""
        return (self.passes_completed / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def progressive_pass_pct(self) -> float:
        """Percentage of passes that were progressive."""
        return (self.progressive_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def progressive_carry_pct(self) -> float:
        """Percentage of carries that were progressive."""
        return (self.progressive_carries / self.carries * 100) if self.carries > 0 else 0.0

    @property
    def tackle_success_rate(self) -> float:
        """Percentage of successful tackles."""
        return (self.tackles_won / self.tackles * 100) if self.tackles > 0 else 0.0

    @property
    def pressure_success_rate(self) -> float:
        """Percentage of successful pressures."""
        return (self.pressure_success / self.pressures * 100) if self.pressures > 0 else 0.0

    @property
    def key_pass_rate(self) -> float:
        """Key passes per 100 passes."""
        return (self.key_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def through_ball_pct(self) -> float:
        """Percentage of passes that were through balls."""
        return (self.through_balls / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def final_third_pass_pct(self) -> float:
        """Percentage of passes into final third."""
        return (self.final_third_passes / self.passes * 100) if self.passes > 0 else 0.0

    @property
    def pressures_per_90(self) -> float:
        """Pressures per 90 minutes."""
        return (self.pressures / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def tackles_per_90(self) -> float:
        """Tackles per 90 minutes."""
        return (self.tackles / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def interceptions_per_90(self) -> float:
        """Interceptions per 90 minutes."""
        return (self.interceptions / self.minutes) * 90 if self.minutes > 0 else 0.0

    @property
    def ball_recoveries_per_90(self) -> float:
        """Ball recoveries per 90 minutes."""
        return (self.ball_recoveries / self.minutes) * 90 if self.minutes > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for display/export."""
        return {
            "player_name": self.player_name,
            "matches": self.matches,
            "minutes": self.minutes,
            "passes": self.passes,
            "passes_completed": self.passes_completed,
            "pass_accuracy": round(self.pass_accuracy, 1),
            "progressive_passes": self.progressive_passes,
            "progressive_pass_pct": round(self.progressive_pass_pct, 1),
            "key_passes": self.key_passes,
            "key_pass_rate": round(self.key_pass_rate, 2),
            "carries": self.carries,
            "progressive_carries": self.progressive_carries,
            "progressive_carry_pct": round(self.progressive_carry_pct, 1),
            "pressures": self.pressures,
            "pressure_success_rate": round(self.pressure_success_rate, 1),
            "tackles": self.tackles,
            "tackle_success_rate": round(self.tackle_success_rate, 1),
            "interceptions": self.interceptions,
            "ball_recoveries": self.ball_recoveries,
            "through_balls": self.through_balls,
            "through_ball_pct": round(self.through_ball_pct, 2),
            "final_third_passes": self.final_third_passes,
            "final_third_pass_pct": round(self.final_third_pass_pct, 1),
        }


def calculate_goalkeeper_stats(events: pl.DataFrame) -> GoalkeeperStats:
    """Calculate goalkeeper statistics from StatsBomb events.

    Args:
        events: DataFrame of StatsBomb events for one goalkeeper

    Returns:
        GoalkeeperStats dataclass with computed statistics
    """
    import math

    if len(events) == 0:
        return GoalkeeperStats(
            player_name="Unknown", matches=0, minutes=0,
            saves=0, goals_conceded=0, passes=0, passes_completed=0,
            long_passes=0, short_passes=0, high_passes=0, total_pass_distance=0,
        )

    # Get player name
    player_name = ""
    if "player" in events.columns:
        first_player = events["player"].drop_nulls().first()
        player_name = first_player if first_player else "Unknown"

    # Count matches
    matches = 0
    if "match_id" in events.columns:
        matches = events.select(pl.col("match_id").n_unique()).item()

    # Estimate minutes
    minutes = 1.0
    if "minute" in events.columns:
        unique_minutes = events.select(pl.col("minute").n_unique()).item()
        minutes = max(unique_minutes or 1, 1)

    # === SAVES ===
    saves = 0
    goals_conceded = 0
    if "type" in events.columns:
        gk_events = events.filter(pl.col("type") == "Goal Keeper")
        if "goalkeeper_type" in events.columns:
            saves = len(gk_events.filter(
                pl.col("goalkeeper_type").is_in(["Save", "Collected", "Punch", "Keeper Sweeper"])
            ))
        # Goals conceded - check for shot events against (trickier, estimate from Goal Keeper events)
        # Approximate: count "Goal Conceded" type events or infer from match context
        if "goalkeeper_outcome" in events.columns:
            goals_conceded = len(gk_events.filter(
                pl.col("goalkeeper_outcome") == "Goal Conceded"
            ))

    # === DISTRIBUTION (PASSES) ===
    passes = 0
    passes_completed = 0
    long_passes = 0
    short_passes = 0
    high_passes = 0
    total_pass_distance = 0.0

    if "type" in events.columns:
        passes_df = events.filter(pl.col("type") == "Pass")
        passes = len(passes_df)

        if passes > 0:
            # Successful passes
            if "pass_outcome" in events.columns:
                passes_completed = len(passes_df.filter(pl.col("pass_outcome").is_null()))
            else:
                passes_completed = passes

            # High passes (pass_height)
            if "pass_height" in events.columns:
                high_passes = len(passes_df.filter(
                    pl.col("pass_height").is_in(["High Pass", "Lofted Pass"])
                ))

            # Calculate pass distances and categorize
            if "location" in events.columns and "pass_end_location" in events.columns:
                try:
                    for row in passes_df.to_dicts():
                        loc = row.get("location")
                        end_loc = row.get("pass_end_location")
                        if loc and end_loc and len(loc) >= 2 and len(end_loc) >= 2:
                            dist = math.sqrt((end_loc[0] - loc[0])**2 + (end_loc[1] - loc[1])**2)
                            total_pass_distance += dist
                            if dist > 32:  # Long pass > 32m
                                long_passes += 1
                            elif dist < 20:  # Short pass < 20m
                                short_passes += 1
                except Exception:
                    pass

    return GoalkeeperStats(
        player_name=player_name,
        matches=matches,
        minutes=minutes,
        saves=saves,
        goals_conceded=goals_conceded,
        passes=passes,
        passes_completed=passes_completed,
        long_passes=long_passes,
        short_passes=short_passes,
        high_passes=high_passes,
        total_pass_distance=total_pass_distance,
    )


def calculate_midfielder_stats(events: pl.DataFrame) -> MidfielderStats:
    """Calculate midfielder statistics from StatsBomb events.

    Args:
        events: DataFrame of StatsBomb events for one midfielder

    Returns:
        MidfielderStats dataclass with computed statistics
    """
    if len(events) == 0:
        return MidfielderStats(
            player_name="Unknown", matches=0, minutes=0,
            passes=0, passes_completed=0, progressive_passes=0, key_passes=0,
            carries=0, progressive_carries=0,
            pressures=0, pressure_success=0, tackles=0, tackles_won=0,
            interceptions=0, ball_recoveries=0, through_balls=0, final_third_passes=0,
        )

    # Get player name
    player_name = ""
    if "player" in events.columns:
        first_player = events["player"].drop_nulls().first()
        player_name = first_player if first_player else "Unknown"

    # Count matches
    matches = 0
    if "match_id" in events.columns:
        matches = events.select(pl.col("match_id").n_unique()).item()

    # Estimate minutes
    minutes = 1.0
    if "minute" in events.columns:
        unique_minutes = events.select(pl.col("minute").n_unique()).item()
        minutes = max(unique_minutes or 1, 1)

    # === PASSES ===
    passes = 0
    passes_completed = 0
    progressive_passes = 0
    key_passes = 0
    through_balls = 0
    final_third_passes = 0

    if "type" in events.columns:
        passes_df = events.filter(pl.col("type") == "Pass")
        passes = len(passes_df)

        if passes > 0:
            # Successful passes
            if "pass_outcome" in events.columns:
                passes_completed = len(passes_df.filter(pl.col("pass_outcome").is_null()))
            else:
                passes_completed = passes

            # Key passes (shot assists)
            if "pass_shot_assist" in events.columns:
                key_passes = len(passes_df.filter(pl.col("pass_shot_assist") == True))

            # Through balls
            if "pass_technique" in events.columns:
                through_balls = len(passes_df.filter(
                    pl.col("pass_technique") == "Through Ball"
                ))

            # Progressive passes and final third passes
            if "location" in events.columns and "pass_end_location" in events.columns:
                try:
                    for row in passes_df.to_dicts():
                        loc = row.get("location")
                        end_loc = row.get("pass_end_location")
                        if loc and end_loc and len(loc) >= 2 and len(end_loc) >= 2:
                            # Progressive = moved ball forward by 10+ meters
                            if end_loc[0] - loc[0] >= 10:
                                progressive_passes += 1
                            # Final third = pass ending in x >= 80 (StatsBomb coordinates)
                            if end_loc[0] >= 80:
                                final_third_passes += 1
                except Exception:
                    pass

    # === CARRIES ===
    carries = 0
    progressive_carries = 0

    if "type" in events.columns:
        carry_df = events.filter(pl.col("type") == "Carry")
        carries = len(carry_df)

        if "carry_end_location" in events.columns and carries > 0:
            try:
                for row in carry_df.to_dicts():
                    loc = row.get("location")
                    end_loc = row.get("carry_end_location")
                    if loc and end_loc and len(loc) >= 2 and len(end_loc) >= 2:
                        # Progressive carry = advanced ball forward by 10+ meters
                        if end_loc[0] - loc[0] >= 10:
                            progressive_carries += 1
            except Exception:
                pass

    # === PRESSURES ===
    pressures = 0
    pressure_success = 0

    if "type" in events.columns:
        pressure_df = events.filter(pl.col("type") == "Pressure")
        pressures = len(pressure_df)

        if "counterpress" in events.columns and pressures > 0:
            pressure_success = len(pressure_df.filter(pl.col("counterpress") == True))

    # === TACKLES ===
    tackles = 0
    tackles_won = 0

    if "type" in events.columns:
        tackles_df = events.filter(pl.col("type") == "Duel")
        if "duel_type" in events.columns:
            tackles_df = tackles_df.filter(pl.col("duel_type") == "Tackle")
        tackles = len(tackles_df)

        if "duel_outcome" in events.columns and tackles > 0:
            tackles_won = len(tackles_df.filter(
                pl.col("duel_outcome").is_in(["Won", "Success", "Success In Play"])
            ))

    # === INTERCEPTIONS ===
    interceptions = 0
    if "type" in events.columns:
        interceptions = len(events.filter(pl.col("type") == "Interception"))

    # === BALL RECOVERIES ===
    ball_recoveries = 0
    if "type" in events.columns:
        ball_recoveries = len(events.filter(pl.col("type") == "Ball Recovery"))

    return MidfielderStats(
        player_name=player_name,
        matches=matches,
        minutes=minutes,
        passes=passes,
        passes_completed=passes_completed,
        progressive_passes=progressive_passes,
        key_passes=key_passes,
        carries=carries,
        progressive_carries=progressive_carries,
        pressures=pressures,
        pressure_success=pressure_success,
        tackles=tackles,
        tackles_won=tackles_won,
        interceptions=interceptions,
        ball_recoveries=ball_recoveries,
        through_balls=through_balls,
        final_third_passes=final_third_passes,
    )
