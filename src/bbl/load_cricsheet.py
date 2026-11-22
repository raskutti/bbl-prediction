"""Parse Cricsheet BBL ball-by-ball JSON into tidy tables.

Usage: python load_cricsheet.py data/bbl_json      (folder of .json files, or a .zip)
Writes data/matches.csv, data/innings.csv, data/balls.csv.gz
Download: https://cricsheet.org/downloads/bbl_json.zip
"""

import json, sys, zipfile, os, glob
import pandas as pd

from bbl.config import DATA_DIR


def parse_match(mid, d):
    info = d["info"]
    teams = info["teams"]
    out = info.get("outcome", {})
    winner = out.get("winner")
    m = dict(
        match_id=mid,
        date=pd.to_datetime(info["dates"][0]),
        season=str(info.get("season")),
        venue=info.get("venue"),
        city=info.get("city"),
        gender=info.get("gender"),
        team_a=teams[0],
        team_b=teams[1],
        toss_winner=info.get("toss", {}).get("winner"),
        toss_decision=info.get("toss", {}).get("decision"),
        winner=winner,
        no_result=int(winner is None),
        method=out.get("method"),  # D/L etc: exclude reduced-over games from the 'full match' models
        match_number=info.get("event", {}).get("match_number"),
        stage=info.get("event", {}).get("stage"),
    )
    innings_rows, ball_rows = [], []
    for i, inn in enumerate(d.get("innings", []), start=1):
        if inn.get("super_over"):
            continue
        team = inn["team"]
        opp = teams[1] if team == teams[0] else teams[0]
        pp_runs = pp_wk = tot = wk = legal = 0
        for ov in inn["overs"]:
            for dl in ov["deliveries"]:
                r = dl["runs"]["total"]
                w = len(dl.get("wickets", []))
                ex = dl.get("extras", {})
                is_legal = not ("wides" in ex or "noballs" in ex)
                tot += r
                wk += w
                legal += is_legal
                if ov["over"] < 6:
                    pp_runs += r
                    pp_wk += w
                ball_rows.append(
                    dict(
                        match_id=mid,
                        innings=i,
                        over=ov["over"],
                        batter=dl["batter"],
                        bowler=dl["bowler"],
                        runs_batter=dl["runs"]["batter"],
                        runs_total=r,
                        wicket=w,
                        legal=int(is_legal),
                    )
                )
        innings_rows.append(
            dict(
                match_id=mid,
                innings=i,
                bat_team=team,
                bowl_team=opp,
                pp_runs=pp_runs,
                pp_wkts=pp_wk,
                total=tot,
                wkts=wk,
                legal_balls=legal,
                overs_bowled=legal / 6,
                full_pp=int(any(o["over"] == 5 for o in inn["overs"])),
            )
        )
    return m, innings_rows, ball_rows


def _iter_json(path):
    """Yield (match_id, parsed_json) from a folder (searched recursively) or a zip file."""
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
            with open(f, encoding="utf-8") as fh:
                yield os.path.basename(f)[:-5], json.load(fh)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    yield n.split("/")[-1][:-5], json.load(z.open(n))
    else:
        raise SystemExit(f"{path} is neither a folder nor a zip file")


def load(path):
    M, I, B = [], [], []
    for mid, d in _iter_json(path):
        if "info" not in d or "innings" not in d:  # skip non-match json
            continue
        m, i, b = parse_match(mid, d)
        M.append(m)
        I += i
        B += b
    if not M:
        raise SystemExit(f"No match JSON files found in {path}")
    return (pd.DataFrame(M).sort_values("date").reset_index(drop=True), pd.DataFrame(I), pd.DataFrame(B))


if __name__ == "__main__":
    m, i, b = load(sys.argv[1])
    m.to_csv(DATA_DIR / "matches.csv", index=False)
    i.to_csv(DATA_DIR / "innings.csv", index=False)
    b.to_csv(DATA_DIR / "balls.csv.gz", index=False)
    print(len(m), "matches,", len(i), "innings,", len(b), "balls")
