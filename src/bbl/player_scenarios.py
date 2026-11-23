"""Apply the validated player-rating covariates to scenario powerplay line-ups for 1 Jan 2027 (MCG, innings 1).
Base = venue+innings NB (mixture of hl=inf and hl=1, as in powerplay_predict). Player ratings from ALL data to 2025/26 (K=60).
Unseen players (no BBL powerplay history) get 0 adjustment = league mean."""

import io, contextlib, numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from bbl.powerplay_model import *
    from scipy.optimize import minimize
F = np.load(RESULTS_DIR / "player_layer_features.npy")
X1, X2, Y, MU = F
# final coefficients: refit on all out-of-sample rows (same ridge as the backtest); alpha from final model below
ven = venues_for(d)
NOW = PREDICTION_YEAR
b = pd.read_csv(DATA_DIR / "balls.csv.gz")
b = b[b.over < 6]
bat = b.groupby("batter").agg(r=("runs_batter", "sum"), n=("legal", "sum"))
lg = bat.r.sum() / bat.n.sum()
bow = b.groupby("bowler").agg(r=("runs_total", "sum"), n=("legal", "sum"))
lgb = bow.r.sum() / bow.n.sum()
bat["rating"] = (bat.r + 60 * lg) / (bat.n + 60)
bow["rating"] = (bow.r + 60 * lgb) / (bow.n + 60)
bat["runs_per_over_vs_league"] = (bat.rating - lg) * 6
bow["runs_per_over_vs_league"] = (bow.rating - lgb) * 6


def show(tbl, names):
    out = []
    for nm in names:
        out.append(
            (nm, int(tbl.n[nm]), round(tbl.runs_per_over_vs_league[nm], 2))
            if nm in tbl.index
            else (nm, 0, 0.0)
        )
    return pd.DataFrame(out, columns=["player", "PP balls (career BBL)", "rating vs league (runs/over)"])


def feat(tbl, names):
    return float(np.mean([tbl.runs_per_over_vs_league[n] if n in tbl.index else 0.0 for n in names]))


# coefficients fit on all OOS rows with alpha ~ base
al = 0.04


def nll(par):
    eta = np.log(MU) + par[0] * X1 + par[1] * X2
    mu = np.exp(eta)
    r = 1 / al
    return -(
        gammaln(Y + r) - gammaln(r) - gammaln(Y + 1) + r * np.log(r / (r + mu)) + Y * np.log(mu / (r + mu))
    ).sum() + 0.5 * 5.0 * np.sum(np.square(par))


cb, cw = minimize(nll, [0, 0]).x
print(f"coefs (per runs/over deviation): batter {cb:.3f}, bowler {cw:.3f}")


# base mean for innings 1 at MCG
def base(cfg, bat_t, bowl_t):
    kind, hl, lam = cfg
    par = fit(d, kind, hl, lam, NOW, ven)
    row = pd.DataFrame(
        dict(bat_team=[bat_t], bowl_team=[bowl_t], venue=[MCG], innings=[1], pp_runs=[0], year=[NOW])
    )
    mu, a = params(par, row, kind, ven)
    return mu[0], float(np.ravel(a)[0])


mu_b = {
    k: np.mean([base(("venue", hl, (20, 0)), *k)[0] for hl in [1e9, 1.0]])
    for k in [("Melbourne Renegades", "Sydney Sixers"), ("Sydney Sixers", "Melbourne Renegades")]
}
alpha = base(("venue", 1e9, (20, 0)), "Melbourne Renegades", "Sydney Sixers")[1]
print(
    "venue+inn base means (MCG, inn 1):",
    {k[0]: round(v, 2) for k, v in mu_b.items()},
    "alpha",
    round(alpha, 3),
)
# ----------------------------------------------------------------------------------------------
# SCENARIO INPUTS - assembled by hand from public squad news, as at 23 Nov 2026. THESE ARE GUESSES
# about the 1 Jan 2027 powerplay personnel, not announced XIs. Names use Cricsheet's initials format;
# a name with no BBL powerplay history (e.g. newcomers) silently gets a zero adjustment (= league
# average), so the unmatched names are printed below: check them against `data/balls.csv.gz`.
#   Renegades: Bairstow, Bracewell, Mustafizur signed as overseas players; Seifert and O'Neill left
#              for the Sixers; Zampa left for the Hurricanes; Behrendorff is out for the season (injury).
#   Sixers:    Seifert, O'Neill, Jordan and Rocchiccioli signed; Babar, Murphy and Henriques gone;
#              Hughes not confirmed.
#   Excluded:  Steve Smith. He is unavailable because the match falls between the Boxing Day Test
#              (26-30 Dec) and the New Year Test (4-8 Jan) against New Zealand. No other player is
#              excluded as a Test player (the Australia Test squad had not been named at the time).
# Update these from the official team announcements on match day.
# ----------------------------------------------------------------------------------------------
REN_B = dict(  # Renegades powerplay batters
    core=["J Brown", "J Fraser-McGurk"],
    plus_bairstow=["J Brown", "J Fraser-McGurk", "JM Bairstow"],
    plus_jewell=["J Brown", "J Fraser-McGurk", "CP Jewell"],
)
SIX_B = dict(  # Sixers powerplay batters
    philippe_seifert=["JR Philippe", "TL Seifert"],
    plus_davies=["JR Philippe", "TL Seifert", "O Davies"],
    plus_silk=["JR Philippe", "TL Seifert", "JC Silk"],
)
REN_W = dict(  # Renegades powerplay bowlers (what the Sixers face)
    suth_dogg_rogers=["W Sutherland", "B Doggett", "TS Rogers"],
    with_mustafizur=["TS Rogers", "B Doggett", "Mustafizur Rahman"],
)
SIX_W = dict(  # Sixers powerplay bowlers (what the Renegades face)
    abbott_dwar_edwards=["SA Abbott", "BJ Dwarshuis", "J Edwards"],
    with_oneill=["SA Abbott", "BJ Dwarshuis", "F O'Neill"],
    with_jordan=["SA Abbott", "CJ Jordan", "BJ Dwarshuis"],
)
_names = {n for v in {**REN_B, **SIX_B, **REN_W, **SIX_W}.values() for n in v}
print(
    "No BBL powerplay history (get a neutral rating):",
    sorted(n for n in _names if n not in bat.index and n not in bow.index),
)
print("\nBatter ratings:")
print(show(bat, sorted({n for v in {**REN_B, **SIX_B}.values() for n in v})).to_string(index=False))
print("\nBowler ratings:")
print(show(bow, sorted({n for v in {**REN_W, **SIX_W}.values() for n in v})).to_string(index=False))
grid = np.arange(0, 161)


def pmf(mu):
    r = 1 / alpha
    return nbinom.pmf(grid, r, r / (r + mu))


def stats(p):
    c = np.cumsum(p)
    mean = (grid * p).sum()
    q = lambda u: int(np.searchsorted(c, u))
    return mean, q(0.05), q(0.5), q(0.95)


print("\nScenario powerplay means (runs).  adjust = exp(cb*batter_feat + cw*opp_bowler_feat)")
res = {}
for team, key, BS, WS_opp in [
    ("Renegades bat first", ("Melbourne Renegades", "Sydney Sixers"), REN_B, SIX_W),
    ("Sixers bat first", ("Sydney Sixers", "Melbourne Renegades"), SIX_B, REN_W),
]:
    for bn, bl in BS.items():
        for wn, wl in WS_opp.items():
            adj = np.exp(cb * feat(bat, bl) + cw * feat(bow, wl))
            mu = mu_b[key] * adj
            m_, p5, med, p95 = stats(pmf(mu))
            res[(team, bn, wn)] = mu
            print(
                f"  {team:20s} bat={bn:17s} opp bowl={wn:16s} adj x{adj:.3f} -> mean {mu:.1f} median {med} [5%:{p5}, 95%:{p95}]"
            )
mus = {
    t: np.mean([v for k, v in res.items() if k[0] == t]) for t in ["Renegades bat first", "Sixers bat first"]
}
print("\nAverage over scenarios (Smith excluded):", {k: round(v, 1) for k, v in mus.items()})
pd.DataFrame({"runs": grid, **{t.replace(" ", "_"): pmf(mu) for t, mu in mus.items()}}).to_csv(
    RESULTS_DIR / "pp_pmf_squad_adjusted_full.csv", index=False
)

# ---- chart of the squad-adjusted distributions (primary powerplay answer) ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(8, 4.2))
lo, hi = 15, 90
for team, colour in [("Renegades bat first", "#d62839"), ("Sixers bat first", "#e6007e")]:
    ax.plot(
        grid[lo : hi + 1],
        pmf(mus[team])[lo : hi + 1],
        label=f"{team} (mean {mus[team]:.1f})",
        color=colour,
        lw=2,
    )
ax.set(
    xlabel="Runs in first six overs",
    ylabel="Probability",
    title="Powerplay runs, squad-adjusted: Renegades v Sixers, MCG, 1 Jan 2027",
)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "pp_distributions_squad_adjusted.png", dpi=150)
