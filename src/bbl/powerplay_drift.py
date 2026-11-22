import io, contextlib, numpy as np, pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from bbl.powerplay_model import *
# 1) is the season level persistent? season means, sampling error, lag-1 autocorr
sm = d.groupby("season").pp_runs.agg(["mean", "var", "count"])
sm["se2"] = sm["var"] / sm["count"]
mu_ = sm["mean"].mean()
tot = sm["mean"].var()
samp = sm.se2.mean()
print(
    f"season-mean sd {sm['mean'].std():.2f}; sampling sd {np.sqrt(samp):.2f}; implied true between-season sd {np.sqrt(max(tot-samp,0)):.2f}"
)
x = sm["mean"].values - mu_
print("lag-1 autocorr of season means:", round(np.corrcoef(x[:-1], x[1:])[0, 1], 2), "(n=14 pairs; SE ~0.27)")
print("last-5-season pairs lag1:", round(np.corrcoef(x[-6:-1], x[-5:])[0, 1], 2))
# 2) pooled out-of-sample log score by half-life for the league NB, all eval seasons vs last 2 seasons
print("\nleague_NB pooled log score by half-life:   all eval   |  2024/25-25/26 only")
for hl in [0.5, 1, 2, 3, 5, 1e9]:
    c = ("const", hl, (0, 0))
    print(f"  hl={hl:>8}: {pooled(c, ev):.4f}  |  {pooled(c, ['2024/25','2025/26']):.4f}")
# 3) team-lambda sensitivity (venue lam 20, hl=inf): pooled score & spread of team effects in final fit
print("\nfull model, team lambda sensitivity (hl=inf, venue lam 20): pooled OOS log score")
for l2 in [30, 100, 300, 1000]:
    c = ("full", 1e9, (20, l2))
    print(f"  team lam={l2:5d}: {pooled(c, ev):.4f}")
print(
    "  venue+inn only      :",
    round(pooled(("venue", 1e9, (20, 0)), ev), 4),
    "| const:",
    round(pooled(("const", 1e9, (0, 0)), ev), 4),
)
