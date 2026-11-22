import io, contextlib, numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from bbl.powerplay_model import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, X = HOME_TEAM, AWAY_TEAM
NOW = PREDICTION_YEAR
grid = np.arange(0, 161)
A = ("full", 1e9, (20, 1000))
B = ("full", 1.0, (20, 1000))
# ---- OOS score of the 50/50 mixture vs components (pre-season, frozen) ----
mix, pa, pb = [], [], []
for s in ev:
    te = d[sv == s]
    yv = te.pp_runs.values

    def pm(c):
        mu, a = store[(c, s)]
        r = 1 / a
        return nbinom.pmf(yv, r, r / (r + mu))

    mix.append(np.log(0.5 * pm(A) + 0.5 * pm(B)))
    pa.append(np.log(pm(A)))
    pb.append(np.log(pm(B)))
print(
    "OOS log score: A(hl=inf) %.4f | B(hl=1) %.4f | 50/50 mixture %.4f | league_NB %.4f"
    % (
        np.concatenate(pa).mean(),
        np.concatenate(pb).mean(),
        np.concatenate(mix).mean(),
        pooled(("const", 1e9, (0, 0)), ev),
    )
)
# ---- fit on all data ----
ven = venues_for(d)
print("venues with own effect:", len(ven), "| MCG included:", MCG in ven)


def pmf_for(cfg, bat, bowl, innings=1, venue=MCG):
    kind, hl, lam = cfg
    par = fit(d, kind, hl, lam, NOW, ven)
    row = pd.DataFrame(
        dict(bat_team=[bat], bowl_team=[bowl], venue=[venue], innings=[innings], pp_runs=[0], year=[NOW])
    )
    mu, a = params(par, row, kind, ven)
    r = 1 / a[()] if np.ndim(a) else 1 / a
    return nbinom.pmf(grid, r, r / (r + mu[0])), mu[0], a, par


def summ(p):
    c = np.cumsum(p)
    q = lambda u: int(np.searchsorted(c, u))
    mean = (grid * p).sum()
    return dict(
        mean=mean,
        sd=np.sqrt(((grid - mean) ** 2 * p).sum()),
        mode=int(grid[p.argmax()]),
        p5=q(0.05),
        p25=q(0.25),
        median=q(0.5),
        p75=q(0.75),
        p95=q(0.95),
    )


res = {}
for name, bat, bowl in [("Renegades_bat_first", R, X), ("Sixers_bat_first", X, R)]:
    pA, muA, aA, parA = pmf_for(A, bat, bowl)
    pB, muB, aB, parB = pmf_for(B, bat, bowl)
    res[name] = 0.5 * pA + 0.5 * pB
    print(f"{name}: mean A {muA:.2f} (alpha {aA:.3f}) | B {muB:.2f}")
print("\nFinal mixture distributions:")
print(pd.DataFrame({k: summ(v) for k, v in res.items()}).round(2).to_string())
# sensitivity: team shrinkage weaker
for l2 in [100, 300]:
    c = ("full", 1e9, (20, l2))
    a_ = summ(pmf_for(c, R, X)[0])
    b_ = summ(pmf_for(c, X, R)[0])
    print(f"sensitivity team lam={l2}: Renegades mean {a_['mean']:.2f}, Sixers mean {b_['mean']:.2f}")
# coefficient readout (model A)
kind, hl, lam = A
par = fit(d, kind, hl, lam, NOW, ven)
names = (
    ["const", "inn2"]
    + [f"venue:{v}" for v in ven]
    + [f"bat:{t}" for t in teams]
    + [f"bowl:{t}" for t in teams]
)
co = pd.Series(par[: len(names)], index=names)
print(
    "\nModel A multiplicative effects (exp(beta)): innings2 %.3f | MCG %.3f | bat Ren %.3f Six %.3f | bowl Ren %.3f Six %.3f"
    % (
        np.exp(co["inn2"]),
        np.exp(co[f"venue:{MCG}"]),
        np.exp(co[f"bat:{R}"]),
        np.exp(co[f"bat:{X}"]),
        np.exp(co[f"bowl:{R}"]),
        np.exp(co[f"bowl:{X}"]),
    )
)
print(
    "alpha(A) =",
    round(np.exp(par[len(names)]), 3),
    " => sd at mean 46 ~",
    round(np.sqrt(46 + np.exp(par[len(names)]) * 46**2), 1),
)
# ---- save ----
out = pd.DataFrame({"runs": grid, **{k: v for k, v in res.items()}})
out.to_csv(RESULTS_DIR / "pp_pmf_full.csv", index=False)
out[(out.runs >= 15) & (out.runs <= 90)].round(5).to_csv(RESULTS_DIR / "pp_pmf_15_90.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, c in [("Renegades_bat_first", "#d6293e"), ("Sixers_bat_first", "#e5007e")]:
    ax.plot(grid[15:91], res[k][15:91], label=k.replace("_", " "), color=c, lw=2)
ax.set_xlabel("Runs in first six overs")
ax.set_ylabel("Probability")
ax.legend()
ax.set_title("Powerplay runs: Renegades v Sixers, MCG, batting first")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pp_distributions.png", dpi=150)
