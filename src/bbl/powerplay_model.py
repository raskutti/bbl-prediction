"""Powerplay-runs model: NB2 regression with ridge-shrunk team/venue effects and time decay.
Rolling-origin: for each test season, fit on earlier seasons only (pre-season, as in deployment)."""

import numpy as np, pandas as pd, itertools
from scipy.optimize import minimize
from scipy.special import gammaln, digamma
from scipy.stats import nbinom

from bbl.negbin import logpmf, nb_nll
from bbl.common import *

m = load_matches()
inn = pd.read_csv(DATA_DIR / "innings.csv")
d = inn.merge(m[["match_id", "date", "season", "venue", "method", "year"]], on="match_id")
print("innings total", len(d), "| dropped short/D-L:", int(((d.full_pp == 0) | d.method.notna()).sum()))
d = d[(d.full_pp == 1) & d.method.isna()].sort_values(["date", "match_id", "innings"]).reset_index(drop=True)
seasons = sorted(d.season.unique())
POOL_START, EVAL_START = "2013/14", "2016/17"
teams = sorted(set(d.bat_team))
tix = {t: i for i, t in enumerate(teams)}
T = len(teams)
y = d.pp_runs.values.astype(float)
sv = d.season.values
yr = d.year.values


def design(df, kind, venues):
    n = len(df)
    parts = [np.ones((n, 1))]
    pen = [0.0]
    if kind != "const":
        parts.append((df.innings.values == 2).astype(float)[:, None])
        pen.append(0.0)
        V = np.zeros((n, len(venues)))
        for j, v in enumerate(venues):
            V[:, j] = df.venue.values == v
        parts.append(V)
        pen += [1] * len(venues)  # venue penalty group = 1
    if kind == "full":
        B = np.zeros((n, T))
        O = np.zeros((n, T))
        B[np.arange(n), df.bat_team.map(tix).values] = 1
        O[np.arange(n), df.bowl_team.map(tix).values] = 1
        parts += [B, O]
        pen += [2] * (2 * T)  # team penalty group = 2
    return np.hstack(parts), np.array(pen)


def fit(df, kind, hl, lam, now, venues, boot_idx=None):
    if boot_idx is not None:
        df = df.iloc[boot_idx]
    X, pen = design(df, kind, venues)
    yv = df.pp_runs.values.astype(float)
    w = 0.5 ** ((now - df.year.values) / hl)
    w = w / w.mean()  # normalise so lam is comparable across hl
    lam_vec = np.where(pen == 1, lam[0], np.where(pen == 2, lam[1], 0.0))
    x0 = np.zeros(X.shape[1] + 1)
    x0[0] = np.log(yv.mean())
    sol = minimize(nb_nll, x0, args=(X, yv, w, lam_vec), jac=True, method="L-BFGS-B")
    return sol.x


def params(par, df, kind, venues):
    X, _ = design(df, kind, venues)
    k = X.shape[1]
    return np.exp(X @ par[:k]), np.exp(par[k])  # mu, alpha


def venues_for(train):
    vc = train.venue.value_counts()
    return list(vc[vc >= 20].index)


# ---------- empirical weighted histogram benchmark ----------
def emp_logpmf(train, yv, hl, now, bw=2.0):
    w = 0.5 ** ((now - train.year.values) / hl)
    grid = np.arange(0, 200)
    dens = (w[:, None] * np.exp(-0.5 * ((grid[None, :] - train.pp_runs.values[:, None]) / bw) ** 2)).sum(0)
    dens = dens / dens.sum()
    return np.log(dens[yv.astype(int)] + 1e-9)


CONFIGS = (
    [("const", hl, (0, 0)) for hl in [0.5, 1, 2, 3, 5, 1e9]]
    + [("venue", hl, (l, 0)) for hl in [0.5, 1, 2, 3, 5, 1e9] for l in [5, 20]]
    + [
        ("full", hl, (l1, l2))
        for hl in [0.5, 1, 2, 3, 5, 1e9]
        for l1 in [5, 20]
        for l2 in [30, 100, 300, 1000]
    ]
)
EMP = [("emp", hl, None) for hl in [1, 2, 3, 5, 1e9]]
test_seasons = [s for s in seasons if s >= POOL_START]
score = {c: {} for c in CONFIGS + EMP}
store = {}
for s in test_seasons:
    tr = d[sv < s]
    te = d[sv == s]
    now = te.year.min()
    ven = venues_for(tr)
    for c in CONFIGS:
        kind, hl, lam = c
        par = fit(tr, kind, hl, lam, now, ven)
        mu, a = params(par, te, kind, ven)
        lp = logpmf(te.pp_runs.values, mu, a)
        score[c][s] = lp
        store[(c, s)] = (mu, a)
    for c in EMP:
        score[c][s] = emp_logpmf(tr, te.pp_runs.values, c[1], now)
allc = CONFIGS + EMP


def pooled(c, ss):
    return np.concatenate([score[c][s] for s in ss]).mean()


# nested selection per family
fam = {
    "league_NB": [c for c in CONFIGS if c[0] == "const"],
    "NB+venue+inn": [c for c in CONFIGS if c[0] == "venue"],
    "NB+teams+venue+inn": [c for c in CONFIGS if c[0] == "full"],
    "empirical_hist": EMP,
}
ev = [s for s in test_seasons if s >= EVAL_START]
out = {}
picks = {}
for f, cands in fam.items():
    lps = []
    picks[f] = []
    for s in ev:
        past = [x for x in test_seasons if x < s]
        best = max(cands, key=lambda c: pooled(c, past))
        lps.append(score[best][s])
        picks[f].append((s, best))
    out[f] = np.concatenate(lps)
n_ev = len(out["league_NB"])
rng = np.random.default_rng(0)
print(f"\nOut-of-sample mean log score per innings, {ev[0]}-{ev[-1]} (n={n_ev}); higher is better")
base = out["league_NB"]
for f, v in out.items():
    dd = v - base
    bs = np.array([dd[rng.integers(0, n_ev, n_ev)].mean() for _ in range(2000)])
    print(
        f"  {f:22s} {v.mean():.4f}   vs league_NB {dd.mean():+.4f}  [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]"
    )
print("\nNested picks for full model (season, (kind, halflife, (venue lam, team lam))):")
for p in picks["NB+teams+venue+inn"]:
    print("  ", p)

# ---------- calibration of the nested full model: PIT + interval coverage ----------
pit, cov80, cov90, resid_mu = [], [], [], []
rngp = np.random.default_rng(1)
for s, c in picks["NB+teams+venue+inn"]:
    te = d[sv == s]
    mu, a = store[(c, s)]
    r = 1 / a
    p = r / (r + mu)
    yv = te.pp_runs.values
    F = nbinom.cdf(yv, r, p)
    F0 = nbinom.cdf(yv - 1, r, p)
    pit += list(F0 + rngp.random(len(yv)) * (F - F0))
    cov80 += list((yv >= nbinom.ppf(0.1, r, p)) & (yv <= nbinom.ppf(0.9, r, p)))
    cov90 += list((yv >= nbinom.ppf(0.05, r, p)) & (yv <= nbinom.ppf(0.95, r, p)))
print(
    "\nPIT histogram (deciles; flat = calibrated):",
    np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit) * 10,
)
print(f"Central 80% interval coverage {np.mean(cov80):.3f}; 90% coverage {np.mean(cov90):.3f}")
# is the team/opposition signal real? mean predicted sd across innings
mus = np.concatenate([store[(c, s)][0] for s, c in picks["NB+teams+venue+inn"]])
print(
    f"sd of predicted means across innings (full model): {mus.std():.2f} runs;  alpha range {min(store[(c,s)][1] for s,c in picks['NB+teams+venue+inn']):.3f}-{max(store[(c,s)][1] for s,c in picks['NB+teams+venue+inn']):.3f}"
)
import pickle

pickle.dump(dict(picks=picks), open(RESULTS_DIR / "pp_picks.pkl", "wb"))

print("\nPicks league_NB / venue:")
for f in ["league_NB", "NB+venue+inn"]:
    print(f, [(a, b[1]) for a, b in picks[f]])
print("\nPre-season predicted mean vs actual, by season (chosen models):")
rows = []
for s, c in picks["NB+teams+venue+inn"]:
    te = d[sv == s]
    mu = store[(c, s)][0]
    cl = picks["league_NB"][[x[0] for x in picks["league_NB"]].index(s)][1]
    rows.append((s, te.pp_runs.mean(), mu.mean(), store[(cl, s)][0].mean(), cl[1]))
print(
    pd.DataFrame(rows, columns=["season", "actual", "full_pred", "league_pred", "league_hl"])
    .round(2)
    .to_string(index=False)
)
