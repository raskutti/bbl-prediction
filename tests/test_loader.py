"""Tests for Cricsheet parsing (powerplay accounting, legal balls, outcomes)."""

import json
import zipfile

from bbl.load_cricsheet import load, parse_match


def _delivery(runs=1, wide=False, wicket=False):
    d = {
        "batter": "A",
        "bowler": "B",
        "non_striker": "C",
        "runs": {"batter": 0 if wide else runs, "extras": runs if wide else 0, "total": runs},
    }
    if wide:
        d["extras"] = {"wides": runs}
    if wicket:
        d["wickets"] = [{"player_out": "A", "kind": "bowled"}]
    return d


def _match(winner="X", overs=7):
    inn = []
    for o in range(overs):
        ds = [_delivery() for _ in range(6)]
        if o == 0:
            ds.append(_delivery(wide=True))  # an extra (illegal) delivery in over 0
            ds[1] = _delivery(wicket=True)
        inn.append({"over": o, "deliveries": ds})
    outcome = {"winner": winner, "by": {"runs": 5}} if winner else {"result": "no result"}
    return {
        "info": {
            "dates": ["2025-12-20"],
            "season": "2025/26",
            "venue": "Melbourne Cricket Ground",
            "gender": "male",
            "teams": ["X", "Y"],
            "toss": {"winner": "X", "decision": "bat"},
            "outcome": outcome,
            "event": {"name": "Big Bash League", "match_number": 3},
        },
        "innings": [{"team": "X", "overs": inn}, {"team": "Y", "overs": inn}],
    }


def test_powerplay_runs_wickets_and_legal_balls():
    m, innings, balls = parse_match("1", _match())
    first = innings[0]
    assert first["bat_team"] == "X" and first["bowl_team"] == "Y"
    assert first["pp_runs"] == 6 * 6 + 1  # 36 off the bat + the wide
    assert first["pp_wkts"] == 1 and first["full_pp"] == 1
    assert first["legal_balls"] == 7 * 6  # the wide is not a legal ball
    assert len(balls) == 2 * (7 * 6 + 1)  # both innings, including the wide


def test_short_innings_is_flagged():
    _, innings, _ = parse_match("1", _match(overs=5))
    assert innings[0]["full_pp"] == 0  # innings ended before over 6 (index 5) was bowled


def test_no_result_and_winner_flags():
    m, _, _ = parse_match("1", _match(winner=None))
    assert m["no_result"] == 1 and m["winner"] is None
    m, _, _ = parse_match("1", _match(winner="Y"))
    assert m["no_result"] == 0 and m["winner"] == "Y"


def test_load_accepts_zip_and_folder(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    for i in range(3):
        (folder / f"{i}.json").write_text(json.dumps(_match()))
    (folder / "readme.json").write_text(json.dumps({"not": "a match"}))  # must be skipped
    zpath = tmp_path / "bbl.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in folder.glob("[0-9].json"):
            z.write(f, f.name)
    for src in (folder, zpath):
        matches, innings, _ = load(str(src))
        assert len(matches) == 3 and len(innings) == 6
