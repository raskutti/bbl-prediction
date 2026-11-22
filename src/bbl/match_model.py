"""Step 2: match-result model. Causal (pre-match) prediction series for many configs; nested config selection;
season-by-season rolling-origin evaluation."""

import numpy as np, pandas as pd, itertools, json
from scipy.optimize import minimize
from bbl.common import *

m = full_matches(load_matches())
seasons = sorted(m.season.unique())
POOL_START, EVAL_START = "2013/14", "2016/17"
teams = sorted(set(m.team_a) | set(m.team_b))
tix = {t: i for i, t in enumerate(teams)}
print(len(m), "matches;", "home_diff!=0 share:", round((m.home_diff != 0).mean(), 3))
print(
    "home win rate when exactly one side at home:",
    round(m[m.home_diff != 0].assign(w=lambda d: np.where(d.home_diff == 1, d.y, 1 - d.y)).w.mean(), 3),
    "n=",
    (m.home_diff != 0).sum(),
)


# ---------------- Elo (live and frozen pre-season snapshot) ----------------
def run_elo(df, k, regress, init=1500.0):
    r = {t: init for t in teams}
    last = None
    live, frozen = [], []
    snap = dict(r)
    for _, g in df.iterrows():
        if last is not None and g.season != last:
            r = {t: init + (v - init) * (1 - regress) for t, v in r.items()}
        if g.season != last:
            snap = dict(r)
            last = g.season
        live.append(r[g.team_a] - r[g.team_b])
        frozen.append(snap[g.team_a] - snap[g.team_b])
        p = 1 / (1 + 10 ** (-(r[g.team_a] - r[g.team_b]) / 400))
        e = k * (g.y - p)
        r[g.team_a] += e
        r[g.team_b] -= e
    return np.array(live), np.array(frozen), r


ELO_GRID = list(itertools.product([4, 8, 12, 16, 24, 32], [0.0, 0.15, 0.3, 0.5]))
elo_live, elo_frozen, elo_final = {}, {}, {}
for k, rg in ELO_GRID:
    a, b, r = run_elo(m, k, rg)
    elo_live[(k, rg)] = a
    elo_frozen[(k, rg)] = b
    elo_final[(k, rg)] = r
sig = lambda z: 1 / (1 + np.exp(-z))
elo_p = lambda d: 1 / (1 + 10 ** (-d / 400))


# ---------------- Bradley-Terry with time decay, refit every 8 matches ----------------
def fit_bt(idx_a, idx_b, hd, y, w, lam):
    n = len(teams)

    def f(par):
        th, h = par[:n], par[n]
        z = th[idx_a] - th[idx_b] + h * hd
        p = sig(z)
        nll = (
            -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum()
            + lam * (th**2).sum()
            + 0.5 * (h**2)
        )
        r = w * (p - y)
        g = np.zeros(n + 1)
        np.add.at(g, idx_a, r)
        np.add.at(g, idx_b, -r)
        g[:n] += 2 * lam * th
        g[n] = (r * hd).sum() + h
        return nll, g

    return minimize(f, np.zeros(n + 1), jac=True, method="L-BFGS-B").x


ia = m.team_a.map(tix).values
ib = m.team_b.map(tix).values
hd = m.home_diff.values.astype(float)
y = m.y.values
yr = m.year.values
BT_GRID = list(itertools.product([1.0, 2.0, 4.0, 1e9], [1.0, 4.0]))  # half-life years, ridge
bt_pred = {c: np.full(len(m), np.nan) for c in BT_GRID}
bt_home = {}
start = int(np.argmax(m.season.values >= POOL_START))
for c in BT_GRID:
    hl, lam = c
    par = None
    for i in range(start, len(m)):
        if (i - start) % 8 == 0:
            w = 0.5 ** ((yr[i] - yr[:i]) / hl)
            par = fit_bt(ia[:i], ib[:i], hd[:i], y[:i], w, lam)
        bt_pred[c][i] = sig(par[ia[i]] - par[ib[i]] + par[len(teams)] * hd[i])
    bt_home[c] = par[len(teams)]
print("BT home coef (last fit, per config):", {k: round(v, 2) for k, v in bt_home.items()})

# ---------------- nested selection + stacking ----------------
sv = m.season.values
ev_seasons = [s for s in seasons if s >= EVAL_START]


def pool_mask(s):
    return (sv >= POOL_START) & (sv < s)


def choose(cands, s):
    pm = pool_mask(s)
    return min(cands, key=lambda c: ll(cands[c][pm], y[pm]).mean())


elo_cands_live = {c: elo_p(elo_live[c]) for c in ELO_GRID}
elo_cands_frozen = {c: elo_p(elo_frozen[c]) for c in ELO_GRID}


def fit_stack(d, hdv, yy):  # symmetric logistic, no intercept: z = b*d/400*ln10 + c*home_diff
    f = lambda p: (
        ll(sig(p[0] * d * np.log(10) / 400 + p[1] * hdv), yy).sum()
        + 0.5 * ((p[0] - 1) ** 2) * 5
        + 0.5 * p[1] ** 2 * 5
    )
    return minimize(f, [1.0, 0.0], method="Nelder-Mead").x


res = {
    k: np.full(len(m), np.nan)
    for k in [
        "coin",
        "elo_live",
        "elo_frozen",
        "elo_live+home",
        "elo_frozen+home",
        "bt",
        "home_only",
        "ensemble",
    ]
}
chosen = []
for s in ev_seasons:
    te = sv == s
    pm = pool_mask(s)
    cl = choose(elo_cands_live, s)
    cf = choose(elo_cands_frozen, s)
    cb = choose(bt_pred, s)
    chosen.append((s, cl, cf, cb))
    res["coin"][te] = 0.5
    res["elo_live"][te] = elo_cands_live[cl][te]
    res["elo_frozen"][te] = elo_cands_frozen[cf][te]
    res["bt"][te] = bt_pred[cb][te]
    for nm, d_all in [("elo_live+home", elo_live[cl]), ("elo_frozen+home", elo_frozen[cf])]:
        b, c = fit_stack(d_all[pm], hd[pm], y[pm])
        res[nm][te] = sig(b * d_all[te] * np.log(10) / 400 + c * hd[te])
    res["home_only"][te] = sig(fit_stack(np.zeros(pm.sum()), hd[pm], y[pm])[1] * hd[te])
    res["ensemble"][te] = 0.5 * res["elo_live+home"][te] + 0.5 * res["bt"][te]
print("\nNested choices (season, Elo live (K,regress), Elo frozen, BT (halflife, ridge)):")
for c in chosen:
    print("  ", c)

E = np.isin(sv, ev_seasons)
yE = y[E]
rng = np.random.default_rng(0)
rows = []
for k, p in res.items():
    pe = p[E]
    L = ll(pe, yE)
    d = L - ll(np.full(E.sum(), 0.5), yE)
    bs = [rng.choice(len(L), len(L)).tolist() for _ in range(2000)]
    dm = np.array([d[b].mean() for b in bs])
    rows.append(
        dict(
            model=k,
            n=E.sum(),
            logloss=L.mean(),
            brier=((pe - yE) ** 2).mean(),
            acc=((pe > 0.5) == (yE == 1)).mean(),
            d_vs_coin=d.mean(),
            ci_lo=np.percentile(dm, 2.5),
            ci_hi=np.percentile(dm, 97.5),
            sd_pred=pe.std(),
        )
    )
tab = pd.DataFrame(rows).set_index("model").round(4)
print("\nOut-of-sample, seasons", ev_seasons[0], "-", ev_seasons[-1])
print(tab.to_string())
tab.to_csv(RESULTS_DIR / "match_eval.csv")
# first ~40% of each season (closest to deployment: Jan 1 is early/mid season)
early = np.zeros(len(m), bool)
for s in ev_seasons:
    idx = np.where(sv == s)[0]
    early[idx[: int(0.4 * len(idx))]] = True
print("\nEarly-season (first 40% of each season) log loss:")
for k, p in res.items():
    print(f"  {k:16s} {ll(p[early], y[early]).mean():.4f}  (n={early.sum()})")

# calibration of ensemble & elo_live+home
for k in ["elo_live+home", "ensemble"]:
    pe = res[k][E]
    q = pd.qcut(pe, 4, duplicates="drop")
    print(f"\nCalibration {k} (quartiles of predicted prob):")
    print(
        pd.DataFrame(dict(pred=pe, y=yE))
        .groupby(q, observed=True)
        .agg(pred=("pred", "mean"), actual=("y", "mean"), n=("y", "size"))
        .round(3)
        .to_string()
    )
pd.DataFrame({"match_id": m.match_id, "season": m.season, "y": y, **{k: v for k, v in res.items()}}).to_csv(
    RESULTS_DIR / "match_preds.csv", index=False
)
json.dump({"chosen": [[str(x) for x in c] for c in chosen]}, open(RESULTS_DIR / "match_chosen.json", "w"))

# ---------------- chase / toss structure ----------------
print("\nChase win rate by era:")
inn = pd.read_csv(DATA_DIR / "innings.csv")
i2 = inn[inn.innings == 2].merge(m[["match_id", "winner", "season", "venue"]], on="match_id")
i2["cw"] = i2.bat_team == i2.winner
print(
    i2.assign(era=np.where(i2.season >= "2021/22", "2021/22+", "earlier"))
    .groupby("era")
    .cw.agg(["mean", "count"])
    .round(3)
    .to_string()
)
mc = i2[i2.venue == MCG]
n, k = len(mc), mc.cw.sum()
print(
    f"MCG chase: {k}/{n} = {k/n:.3f}; league ex-MCG {i2[i2.venue!=MCG].cw.mean():.3f}; SE {np.sqrt(.25/n):.3f}"
)
print(
    "MCG chase last 5 seasons:",
    i2[(i2.venue == MCG) & (i2.season >= "2021/22")].cw.agg(["mean", "count"]).round(3).to_dict(),
)
