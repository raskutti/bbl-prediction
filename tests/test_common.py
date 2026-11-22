"""Tests for shared helpers: venue -> city mapping, home flags, match filtering."""

import numpy as np
import pandas as pd

from bbl.common import TEAM_CITY, full_matches, ll, venue_city


def test_venue_city_mapping():
    assert venue_city("Melbourne Cricket Ground") == "Melbourne"
    assert venue_city("Marvel Stadium, Melbourne") == "Melbourne"
    assert venue_city("Bellerive Oval, Hobart") == "Hobart"
    assert venue_city("Manuka Oval, Canberra") == "neutral"
    assert venue_city(None) == "neutral"


def test_both_melbourne_clubs_share_a_home_city():
    # Documented modelling choice: the MCG counts as 'home' for both Melbourne clubs.
    assert (
        TEAM_CITY["Melbourne Renegades"]
        == TEAM_CITY["Melbourne Stars"]
        == venue_city("Melbourne Cricket Ground")
    )


def test_full_matches_excludes_no_result_and_dl():
    m = pd.DataFrame({"no_result": [0, 1, 0], "method": [None, None, "D/L"], "winner": ["A", None, "B"]})
    out = full_matches(m)
    assert list(out.winner) == ["A"]


def test_log_loss_is_clipped_and_correct():
    assert np.isclose(ll(0.5, 1.0), np.log(2))
    assert np.isfinite(ll(0.0, 1.0)) and np.isfinite(ll(1.0, 0.0))
