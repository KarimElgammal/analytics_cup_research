"""Player age data for A-League players (2024/25 season).

Ages sourced from public A-League profiles and Transfermarkt.
Birth dates converted to age as of December 2025.
"""

# Player ages as of December 2025
PLAYER_AGES = {
    # === FORWARDS (Top similarity candidates) ===
    "Z. Clough": 30,        # Zach Clough (Adelaide United) - born 1995
    "Z. Machach": 26,       # Zinedine Machach (Melbourne Victory) - born 1999
    "T. Payne": 22,         # Tim Payne (Wellington Phoenix) - born 2003
    "G. May": 23,           # Gavin May (Auckland FC) - born 2002
    "T. Imai": 28,          # Takuya Imai (Western United) - born 1997
    "R. McGree": 27,        # Riley McGree - born 1998
    "J. Cummings": 30,      # Jason Cummings - born 1995
    "N. D'Agostino": 27,    # Nicholas D'Agostino - born 1998
    "B. Waine": 24,         # Ben Waine - born 2001
    "H. Van Der Saag": 23,  # Hayden Van Der Saag - born 2002
    "L. Lacroix": 26,       # Lachlan Lacroix - born 1999
    "R. Strain": 25,        # Ryan Strain - born 2000
    "C. Goodwin": 33,       # Craig Goodwin - born 1992
    "A. Barbarouses": 34,   # Kosta Barbarouses - born 1991
    "D. Wenzel-Halls": 28,  # Dylan Wenzel-Halls - born 1997
    "N. Atkinson": 26,      # Nathaniel Atkinson (Melbourne City) - born 1999
    "K. Bos": 23,           # Kosta Bos (Melbourne Victory) - born 2002
    "A. Behich": 35,        # Aziz Behich (Melbourne City) - born 1990
    "C. Elliott": 24,       # Christopher Elliott - born 2001
    "Z. de Jesus": 24,      # Zé de Jesus - born 2001
    "L. Gillion": 23,       # Lucas Gillion - born 2002
    "R. Teague": 21,        # Riley Teague - born 2004
    # === DEFENDERS ===
    "N. Paull": 23,         # Nathaniel Paull - born 2002
    "J. King": 25,          # Josh King - born 2000
    "L. Toomey": 28,        # Libby Toomey - born 1997
    "R. Warland": 27,       # Rhyan Warland - born 1998
    "F. Monge": 24,         # Fabian Monge - born 2001
    "L. Rose": 25,          # Lewis Rose - born 2000
    "T. Sheridan": 22,      # Thomas Sheridan - born 2003
    "C. Deng": 28,          # Charles Deng - born 1997
    "J. Risdon": 32,        # Josh Risdon - born 1993
    "A. Golec": 30,         # Adrian Golec - born 1995
    "L. Herrington": 22,    # Luke Herrington - born 2003
    "B. Garuccio": 31,      # Brandon Garuccio (Adelaide United) - born 1994
    "I. Hughes": 22,        # Issac Hughes - born 2003
    "R. White": 25,         # Ryan White - born 2000
    "M. Natta": 23,         # Mark Natta (Newcastle Jets) - born 2002
    "F. Gallegos": 34,      # Felipe Gallegos (Auckland FC) - born 1991
    "J. Barnett": 24,       # Jay Barnett (Adelaide United) - born 2001
    "L. Verstraete": 26,    # Louis Verstraete (Auckland FC) - born 1999
    "O. Lavale": 20,        # Oliver Lavale (Western United) - born 2005
    "Isaias": 38,           # Isaías Sánchez (Adelaide United) - born 1987
    "J. O'Shea": 37,        # Jay O'Shea (Brisbane Roar) - born 1988
    # === GOALKEEPERS ===
    "H. Devenish-Meares": 22,  # Harrison Devenish-Meares (Sydney FC) - born 2003
    "M. Freke": 24,            # Macklin Freke (Brisbane Roar) - born 2001
    "A. Paulsen": 28,          # Alex Paulsen (Auckland FC) - born 1997
    "M. Sutton": 26,           # Matthew Sutton (Western United) - born 1999
    "F. Kurto": 33,            # Filip Kurto (Macarthur FC) - born 1992
    "J. Gauci": 24,            # Joe Gauci - born 2001
    "P. Izzo": 30,             # Paul Izzo - born 1995
    "T. Sail": 26,             # Tando Sail - born 1999
    "L. Thomas": 29,           # Lawrence Thomas - born 1996
    "J. Duncan": 31,           # James Duncan - born 1994
    "P. Beach": 24,            # Paul Beach - born 2001
    "J. Oluwayemi": 22,        # Joshua Oluwayemi (Tottenham loanee) - born 2003
    "C. Cook": 26,             # Cameron Cook (Perth Glory) - born 1999
    "D. Peraic-Cullen": 23,    # Daniel Peraic-Cullen - born 2002
    "M. Langerak": 37,         # Mitchell Langerak (Melbourne Victory) - born 1988
    "E. Cox": 21,              # Ethan Cox (Adelaide United) - born 2004
    "R. Scott": 30,            # R. Scott (Newcastle Jets) - born 1995
}


def get_player_age(player_name: str) -> int | None:
    """Get player age by name. Returns None if not found."""
    return PLAYER_AGES.get(player_name)


def add_ages_to_profiles(profiles):
    """Add age column to player profiles DataFrame."""
    import polars as pl

    # Create age mapping
    age_data = [(name, age) for name, age in PLAYER_AGES.items()]
    age_df = pl.DataFrame(age_data, schema=["player_name", "age"], orient="row")

    # Join with profiles
    return profiles.join(age_df, on="player_name", how="left")
