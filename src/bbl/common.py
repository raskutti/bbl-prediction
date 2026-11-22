"""Shared helpers: data loading, full-match filter, home-city flags and log loss."""

import numpy as np, pandas as pd
from bbl.config import *

TEAM_CITY = {
    "Melbourne Renegades": "Melbourne",
    "Melbourne Stars": "Melbourne",
    "Sydney Sixers": "Sydney",
    "Sydney Thunder": "Sydney",
    "Perth Scorchers": "Perth",
    "Brisbane Heat": "Brisbane",
    "Adelaide Strikers": "Adelaide",
    "Hobart Hurricanes": "Hobart",
}


def venue_city(v):
    v = str(v)
    for key, c in [
        ("Melbourne", "Melbourne"),
        ("Marvel", "Melbourne"),
        ("Etihad", "Melbourne"),
        ("Junction Oval", "Melbourne"),
        ("Sydney", "Sydney"),
        ("ANZ Stadium", "Sydney"),
        ("ENGIE", "Sydney"),
        ("Perth", "Perth"),
        ("WACA", "Perth"),
        ("Brisbane", "Brisbane"),
        ("Adelaide", "Adelaide"),
        ("Bellerive", "Hobart"),
        ("University of Tasmania", "Hobart"),
    ]:
        if key in v:
            return c
    return "neutral"  # Canberra, Carrara, Geelong, regional venues, Alice Springs ...


def load_matches():
    m = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
    m = m[m.gender.fillna("male") == "male"].sort_values(["date", "match_id"]).reset_index(drop=True)
    m["vcity"] = m.venue.map(venue_city)
    m["home_a"] = (m.team_a.map(TEAM_CITY) == m.vcity).astype(int)
    m["home_b"] = (m.team_b.map(TEAM_CITY) == m.vcity).astype(int)
    m["home_diff"] = m.home_a - m.home_b
    m["y"] = (m.winner == m.team_a).astype(float)
    m["year"] = m.date.dt.year + (m.date.dt.dayofyear - 1) / 365.25
    return m


def full_matches(m):  # completed, no D/L, decided
    return m[(m.no_result == 0) & (m.method.isna())].reset_index(drop=True)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
