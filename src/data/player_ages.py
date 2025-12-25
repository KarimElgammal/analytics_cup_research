"""Player age data for A-League players (2024/25 season).

Ages sourced from public A-League profiles and Transfermarkt.
Birth dates converted to age as of January 2025.
"""

# Player ages as of January 2025
PLAYER_AGES = {
    # Top similarity candidates
    "Z. Clough": 29,        # Zach Clough (Adelaide United) - born 1995
    "Z. Machach": 25,       # Zinedine Machach (Melbourne Victory) - born 1999
    "T. Payne": 21,         # Tim Payne (Wellington Phoenix) - born 2003
    "G. May": 22,           # Gavin May (Auckland FC) - born 2002
    "T. Imai": 27,          # Takuya Imai (Western United) - born 1997
    "R. McGree": 26,        # Riley McGree - born 1998
    "J. Cummings": 29,      # Jason Cummings - born 1995
    "N. D'Agostino": 26,    # Nicholas D'Agostino - born 1998
    "B. Waine": 23,         # Ben Waine - born 2001
    "H. Van Der Saag": 22,  # Hayden Van Der Saag - born 2002
    "L. Lacroix": 25,       # Lachlan Lacroix - born 1999
    "R. Strain": 24,        # Ryan Strain - born 2000
    "C. Goodwin": 32,       # Craig Goodwin - born 1992
    "A. Barbarouses": 33,   # Kosta Barbarouses - born 1991
    "D. Wenzel-Halls": 27,  # Dylan Wenzel-Halls - born 1997
}


def get_player_age(player_name: str) -> int | None:
    """Get player age by name. Returns None if not found."""
    return PLAYER_AGES.get(player_name)


def add_ages_to_profiles(profiles):
    """Add age column to player profiles DataFrame."""
    import polars as pl

    # Create age mapping
    age_data = [(name, age) for name, age in PLAYER_AGES.items()]
    age_df = pl.DataFrame(age_data, schema=["player_name", "age"])

    # Join with profiles
    return profiles.join(age_df, on="player_name", how="left")
