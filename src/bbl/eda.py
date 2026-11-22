"""EDA + Elo baseline with rolling-origin evaluation. Run after load_cricsheet.py."""

import numpy as np, pandas as pd

from bbl.config import *

m = pd.read_csv(DATA_DIR / "matches.csv", parse_dates=["date"])
inn = pd.read_csv(DATA_DIR / "innings.csv")
m = m[m.gender.fillna("male") == "male"]
full = m[(m.no_result == 0) & (m.method.isna())]  # completed, no D/L
print(f"{len(m)} matches; {len(full)} completed full-length; seasons {m.season.nunique()}")

# ---- EDA ----------------------------------------------------------------
x = inn.merge(m[["match_id", "date", "season", "venue", "method"]], on="match_id")
x = x[(x.full_pp == 1) & x.method.isna()]
print("\nPowerplay runs, all innings:\n", x.pp_runs.describe().round(2).to_string())
print("var/mean ratio:", round(x.pp_runs.var() / x.pp_runs.mean(), 2), "(>1 => overdispersed vs Poisson)")
print("\nPP mean by innings:\n", x.groupby("innings").pp_runs.mean().round(2).to_string())
print(
    "\nPP mean by season:\n", x.groupby("season").pp_runs.agg(["mean", "std", "count"]).round(2).to_string()
)
mcg = x[x.venue.str.contains(MCG, na=False)]
print(f"\nMCG PP runs: n={len(mcg)}, mean={mcg.pp_runs.mean():.1f}, sd={mcg.pp_runs.std():.1f}")
for t in ["Melbourne Renegades", "Sydney Sixers"]:
    for side, col in [("batting", "bat_team"), ("bowling vs", "bowl_team")]:
        s = x[x[col] == t]
        print(
            f"{t:20s} {side:10s} PP mean {s.pp_runs.mean():.1f} (n={len(s)}, last 3 seasons "
            f"{s[s.season.isin(sorted(x.season.unique())[-3:])].pp_runs.mean():.1f})"
        )

# chasing advantage: did the team batting second win?
inn2 = inn[inn.innings == 2].merge(full[["match_id", "winner", "venue"]], on="match_id")
inn2["chase_win"] = inn2.bat_team == inn2.winner
print(f"\nChasing win rate overall: {inn2.chase_win.mean():.3f} (n={len(inn2)})")
print(inn2.groupby("venue").chase_win.agg(["mean", "count"]).query("count>=10").round(3).to_string())


# ---- Elo baseline, expanding window ----------------------------------------
def run_elo(df, k, regress, adv=0.0, init=1500):
    r, last_season, rows = {}, None, []
    for _, g in df.iterrows():
        if last_season is not None and g.season != last_season:  # shrink toward mean between seasons
            r = {t: init + (v - init) * (1 - regress) for t, v in r.items()}
        last_season = g.season
        ra, rb = r.get(g.team_a, init), r.get(g.team_b, init)
        p = 1 / (1 + 10 ** (-(ra - rb) / 400))
        y = float(g.winner == g.team_a)
        rows.append((g.match_id, g.season, p, y))
        r[g.team_a] = ra + k * (y - p)
        r[g.team_b] = rb - k * (y - p)
    return pd.DataFrame(rows, columns=["match_id", "season", "p", "y"]), r


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


seasons = sorted(full.season.unique())
burn = seasons[2:]  # score only after 2 burn-in seasons
print("\nElo grid (log loss on post-burn-in matches; 0.693 = coin flip):")
best = None
for k in [5, 10, 15, 20, 30]:
    for rg in [0.0, 0.2, 0.4, 0.6]:
        pr, _ = run_elo(full, k, rg)
        s = pr[pr.season.isin(burn)]
        ll = logloss(s.p.values, s.y.values)
        if best is None or ll < best[0]:
            best = (ll, k, rg)
        print(f"  K={k:2d} regress={rg:.1f}  ll={ll:.4f}")
print("best:", best, "| coin-flip:", round(np.log(2), 4))
# NB: grid-picking on the same data you score is optimistic -> nest this in the real backtest.
_, ratings = run_elo(full, best[1], best[2])
for t in ["Melbourne Renegades", "Sydney Sixers"]:
    print(f"{t}: Elo {ratings.get(t, np.nan):.0f}")
rr, ss = ratings.get("Melbourne Renegades"), ratings.get("Sydney Sixers")
print("Elo-only P(Renegades win):", round(1 / (1 + 10 ** (-(rr - ss) / 400)), 3))
