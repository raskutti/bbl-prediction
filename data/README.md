# Data

Nothing in this folder is committed (see `.gitignore`); everything is re-created from public sources.

| Source | What | How |
|---|---|---|
| [Cricsheet](https://cricsheet.org/downloads/) `bbl_json.zip` | Ball-by-ball men's BBL, 2011/12 to the end of 2025/26 (662 matches) | `make data` downloads it into `data/raw/` |

`make data` then runs `python -m bbl.load_cricsheet`, which accepts either the zip or a folder of
extracted `.json` files, and writes three tidy tables here:

| File | Grain | Key columns |
|---|---|---|
| `matches.csv` | one row per match | `match_id, date, season, venue, team_a, team_b, toss_*, winner, no_result, method` |
| `innings.csv` | one row per innings | `bat_team, bowl_team, pp_runs, pp_wkts, total, full_pp` (`full_pp = 0` if the innings ended inside six overs) |
| `balls.csv.gz` | one row per delivery | `batter, bowler, over, runs_batter, runs_total, wicket, legal` |

Please read Cricsheet's terms of use and attribution requirements before redistributing the data
or anything derived from it. The 2026/27 season had not started when this was written (the data ends
25 Jan 2026), so no current-season games are in the model.
