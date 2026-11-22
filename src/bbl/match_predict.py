"""Deployed win probability: Renegades v Sixers at MCG (pre-toss), ratings through 2025/26."""

import numpy as np, pandas as pd, sys, io, contextlib

with contextlib.redirect_stdout(io.StringIO()):
    import bbl.match_model as S  # re-uses all functions/series (re-runs backtest silently)
from bbl.match_model import *

R, X = HOME_TEAM, AWAY_TEAM
# --- Elo: choose config on ALL history (causal pool), take season-start snapshot for 2026/27 ---
pm = sv >= POOL_START
cf = min(elo_cands_frozen, key=lambda c: ll(elo_cands_frozen[c][pm], y[pm]).mean())
k, rg = cf
r = dict(elo_final[cf])
snap = {t: 1500 + (v - 1500) * (1 - rg) for t, v in r.items()}
b, c = fit_stack(elo_frozen[cf][pm], hd[pm], y[pm])
print(f"Elo frozen config K={k}, regress={rg}; stack b={b:.2f} (1=raw Elo), home logit c={c:.2f}")
print("Elo ratings entering 2026/27:", {t: round(v) for t, v in sorted(snap.items(), key=lambda x: -x[1])})


def p_elo(home):
    return sig(b * (snap[R] - snap[X]) * np.log(10) / 400 + c * home)


# --- BT: choose config over all history, fit on all matches, time-decayed from now ---
cb = min(bt_pred, key=lambda c_: ll(bt_pred[c_][pm], y[pm]).mean())
hl, lam = cb
now = PREDICTION_YEAR  # decay is measured to the fixture date


def bt_fit(idx):
    w = 0.5 ** ((now - yr[idx]) / hl)
    return fit_bt(ia[idx], ib[idx], hd[idx], y[idx], w, lam)


par = bt_fit(np.arange(len(m)))
n = len(teams)
print(f"BT config half-life={hl}y ridge={lam}; home logit={par[n]:.2f}")
print("BT strengths:", {t: round(par[tix[t]], 2) for t in sorted(teams, key=lambda t: -par[tix[t]])})


def p_bt(par, home):
    return sig(par[tix[R]] - par[tix[X]] + par[n] * home)


print("\nP(Renegades win)  [Renegades at home vs Sydney team, MCG = Melbourne]:")
for home in [1, 0]:
    pe, pb = p_elo(home), p_bt(par, home)
    print(f"  home_diff={home:+d}: Elo-stack {pe:.3f} | BT {pb:.3f} | ensemble {(pe+pb)/2:.3f}")
rng = np.random.default_rng(1)
bs = []
for _ in range(300):
    idx = np.sort(rng.integers(0, len(m), len(m)))
    bs.append(p_bt(bt_fit(idx), 1))
print(
    "BT bootstrap 90% interval (home=+1):",
    np.round(np.percentile(bs, [5, 95]), 3),
    "sd",
    round(np.std(bs), 3),
)
# toss: pre-toss averaging over who chases
z = np.log(((p_elo(1) + p_bt(par, 1)) / 2) / (1 - (p_elo(1) + p_bt(par, 1)) / 2))
for cc in [0.2, 0.47]:
    print(
        f"chase logit effect c={cc}: pre-toss p = {0.5*(sig(z+cc)+sig(z-cc)):.3f} vs ignoring chase {sig(z):.3f}"
    )
# recent record
last = m[m.season >= "2024/25"]
for t in [R, X]:
    g = last[(last.team_a == t) | (last.team_b == t)]
    w = ((g.winner == t)).sum()
    print(
        f"{t}: {w}/{len(g)} wins since 2024/25;  last season:",
        ((m[(m.season == "2025/26") & ((m.team_a == t) | (m.team_b == t))].winner == t).sum()),
        "/",
        len(m[(m.season == "2025/26") & ((m.team_a == t) | (m.team_b == t))]),
    )
h2h = m[((m.team_a == R) & (m.team_b == X)) | ((m.team_a == X) & (m.team_b == R))]
print("H2H all-time Renegades wins:", (h2h.winner == R).sum(), "of", len(h2h))


# --- Sensitivity to roster churn (a JUDGEMENT-BASED check, not a validated correction) ---
# Both clubs have turned over heavily since the last games in the data (Jan 2026), while the ratings
# only know last season. Shrinking the strength gap toward zero says "trust last season's gap less".
# The backtest has not tested any particular shrink factor, so this is reported as a range.
import json  # noqa: E402

print("\nP(Renegades win) when only a fraction of the rating gap is kept (roster-churn sensitivity):")
sens = {}
gap_elo = b * (snap[R] - snap[X]) * np.log(10) / 400
gap_bt = par[tix[R]] - par[tix[X]]
for keep in [1.0, 0.75, 0.5, 0.25]:
    for label, home in [("home", 1), ("neutral", 0)]:
        pe = sig(keep * gap_elo + c * home)
        pb = sig(keep * gap_bt + par[n] * home)
        sens[f"keep{keep:.2f}_{label}"] = round(float((pe + pb) / 2), 3)
        print(
            f"  keep {keep:4.0%} of gap | {label:7s}: Elo {pe:.3f}  BT {pb:.3f}  ensemble {(pe + pb) / 2:.3f}"
        )

json.dump(
    {
        "p_renegades_win_home": round(float((p_elo(1) + p_bt(par, 1)) / 2), 3),
        "p_renegades_win_neutral": round(float((p_elo(0) + p_bt(par, 0)) / 2), 3),
        "bt_bootstrap_90": [round(float(v), 3) for v in np.percentile(bs, [5, 95])],
        "churn_sensitivity": sens,
    },
    open(RESULTS_DIR / "match_prediction.json", "w"),
    indent=2,
)
