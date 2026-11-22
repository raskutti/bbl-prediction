"""Player layer: do player-level ratings add out-of-sample powerplay information beyond the team/venue NB model?
Causal player ratings (empirical-Bayes shrunk PP runs per legal ball, seasons < s only), averaged over the
batters who faced PP balls (and bowlers who bowled PP balls) in each innings = 'XI known' (post-announcement) setting.
Added as covariates on top of the venue+innings NB mean (offset), coefficients by leave-one-season-out CV.
"""

import io, contextlib, numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from bbl.powerplay_model import *
from scipy.optimize import minimize

b = pd.read_csv(DATA_DIR / "balls.csv.gz")
b = b[b.over < 6].merge(
    d[["match_id", "innings", "season", "bat_team", "bowl_team"]],
    on=["match_id", "innings", "season"] if "season" in b else ["match_id", "innings"],
)
K_BAT, K_BOWL = 60, 60  # shrinkage strength in balls (prior sample size)


def ratings(sub, who, runs_col, K):
    g = sub.groupby(who).agg(r=(runs_col, "sum"), n=("legal", "sum"))
    lg = g.r.sum() / g.n.sum()
    return ((g.r + K * lg) / (g.n + K)), lg


rows = []
for s in ev:
    prior = b[b.season < s]
    bat_r, lg = ratings(prior, "batter", "runs_batter", K_BAT)
    # bowler: total runs conceded (incl. extras) per legal ball
    bowl_r, lgb = ratings(prior, "bowler", "runs_total", K_BOWL)
    rows.append((s, bat_r, lg, bowl_r, lgb))
RAT = {s: (br, lg, wr, lgb) for s, br, lg, wr, lgb in rows}
# per-innings features
cur = b.groupby(["match_id", "innings"])
bat_players = cur.batter.unique()
bowl_players = cur.bowler.unique()


def feat(s, r):
    br, lg, wr, lgb = RAT[s]
    bp = bat_players.get((r.match_id, r.innings), [])
    wp = bowl_players.get((r.match_id, r.innings), [])
    fb = np.mean([br.get(p, lg) for p in bp]) - lg if len(bp) else 0.0  # unseen player => league mean (0)
    fw = np.mean([wr.get(p, lgb) for p in wp]) - lgb if len(wp) else 0.0
    return fb * 6, fw * 6  # scale to 'runs per over' deviations


# base model: nested pick of NB+venue+inn per season (same selection as powerplay_model)
base_pick = dict(picks["NB+venue+inn"])
lp0, X1, X2, Y, MU, AL, SEAS = [], [], [], [], [], [], []
for s in ev:
    te = d[sv == s].reset_index(drop=True)
    mu, al = store[(base_pick[s], s)]
    F = np.array([feat(s, r) for r in te.itertuples()])
    X1 += list(F[:, 0])
    X2 += list(F[:, 1])
    Y += list(te.pp_runs.values)
    MU += list(mu)
    AL += [al] * len(te)
    SEAS += [s] * len(te)
X1, X2, Y, MU, AL, SEAS = map(np.array, (X1, X2, Y, MU, AL, SEAS))
print(
    f"n innings {len(Y)}; corr(actual/base_mu, batter feat) = {np.corrcoef(Y/MU, X1)[0,1]:.3f}; vs bowler feat = {np.corrcoef(Y/MU, X2)[0,1]:.3f}"
)


def nll(par, idx, cols):
    eta = np.log(MU[idx]) + sum(p * c[idx] for p, c in zip(par, cols))
    mu = np.exp(eta)
    r = 1 / AL[idx]
    return -(
        gammaln(Y[idx] + r)
        - gammaln(r)
        - gammaln(Y[idx] + 1)
        + r * np.log(r / (r + mu))
        + Y[idx] * np.log(mu / (r + mu))
    ).sum() + 0.5 * 5.0 * np.sum(np.square(par))


def loso(cols):
    lp = np.zeros(len(Y))
    coefs = []
    for s in ev:
        te = SEAS == s
        tr = ~te
        par = minimize(nll, np.zeros(len(cols)), args=(np.where(tr)[0], cols)).x if cols else np.array([])
        eta = np.log(MU[te]) + sum(p * c[te] for p, c in zip(par, cols))
        mu = np.exp(eta)
        r = 1 / AL[te]
        lp[te] = nbinom.logpmf(Y[te], r, r / (r + mu))
        coefs.append(par)
    return lp, np.array(coefs)


rng = np.random.default_rng(0)
res = {}
for name, cols in [("base (venue+inn)", []), ("+batter", [X1]), ("+bowler", [X2]), ("+both", [X1, X2])]:
    lp, co = loso(cols)
    res[name] = lp
    dd = lp - res["base (venue+inn)"]
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
    print(
        f"{name:18s} logscore {lp.mean():.4f}  vs base {dd.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]",
        "  mean coefs",
        np.round(co.mean(0), 3) if len(co) and co.size else "",
    )
np.save(RESULTS_DIR / "player_layer_features.npy", np.vstack([X1, X2, Y, MU]))
