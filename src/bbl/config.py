"""Project-wide configuration: file locations and the fixture being predicted.

Everything fixture-specific lives here so the pipeline can be re-pointed at another
match by editing one file. Paths can be overridden with BBL_DATA_DIR / BBL_RESULTS_DIR.
"""

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "HOME_TEAM",
    "AWAY_TEAM",
    "VENUE",
    "MCG",
    "FIXTURE_DATE",
    "PREDICTION_YEAR",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BBL_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("BBL_RESULTS_DIR", REPO_ROOT / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- The fixture: Melbourne Renegades v Sydney Sixers, MCG, Fri 1 Jan 2027, 4.00pm local ----
# The MCG is the Renegades' new home ground for BBL|16 (this is their first home game there),
# so the Renegades are treated as the home side. This is a modelling assumption worth stating.
HOME_TEAM = "Melbourne Renegades"
AWAY_TEAM = "Sydney Sixers"
VENUE = "Melbourne Cricket Ground"
MCG = VENUE  # alias used throughout
FIXTURE_DATE = "2027-01-01"
# Fixture date as a decimal year; time-decay weights are measured back from here.
PREDICTION_YEAR = 2027.0
