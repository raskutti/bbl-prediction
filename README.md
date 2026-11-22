# BBL match prediction: Melbourne Renegades v Sydney Sixers

Pre-toss forecasts for the men's Big Bash League match at the **MCG, Friday 1 January 2027, 4.00pm**,
assuming a full 20-over match:

1. the probability the Renegades win, and
2. the distribution of runs in the **first six overs** for each side batting first.

Built for the Outside Edge data-science take-home. Everything below is reproducible from public data
with `make install && make data && make all` (about 40 seconds).

![Powerplay distributions](results/pp_distributions_squad_adjusted.png)

## Predictions

| | Value | Notes |
|---|---|---|
| **P(Renegades win)** | **0.43** (plausible range 0.38 to 0.50) | Model output is **0.41**; the headline is nudged up slightly for roster churn (see below). The 0.43 is a judgement call, not a backtested number. |
| Renegades powerplay, batting first | mean 44.3, sd 11.3, median 44, 90% range 27 to 64 | Squad-adjusted (primary) |
| Sixers powerplay, batting first | mean 45.1, sd 11.5, median 44, 90% range 28 to 65 | Squad-adjusted (primary) |

Full probability mass functions (0 to 160 runs): [`results/pp_pmf_squad_adjusted_full.csv`](results/pp_pmf_squad_adjusted_full.csv).

**Stated assumptions**

- Full 20-over match, no rain reduction, no abandonment.
- The Renegades are the **home** side (the MCG is their new home ground for BBL|16 and this is their first game there). Treated as neutral, the win probability would be about 0.35.
- Ratings come from games up to January 2026. **No 2026/27 games exist yet**, so squads enter only through the powerplay player layer and the churn sensitivity.
- Powerplay personnel are scenarios, not announced XIs (see [`player_scenarios.py`](src/bbl/player_scenarios.py)). Steve Smith is assumed unavailable (a single-source report); with him, the Sixers' mean would be about 47 to 48.
- The toss is marginalised: it moves the pre-toss win probability by under half a point.

## Method in brief

### Match result ([`match_model.py`](src/bbl/match_model.py), [`match_predict.py`](src/bbl/match_predict.py))

- Two rating models fitted on completed, full-length matches: **Elo** (grid over K and between-season regression) and a **Bradley-Terry** model with a ridge penalty, exponential time decay and a home term.
- "Home" means the club's home city, so for the Melbourne clubs the MCG counts as home for both (a documented simplification).
- Elo and Bradley-Terry are combined 50/50. Hyperparameters are chosen **nested**: for each test season, using only earlier seasons.
- Backtest: rolling origin, 469 matches from 2016/17 to 2025/26, scored on log loss and Brier, with a bootstrap against a coin flip.

### Powerplay runs ([`powerplay_model.py`](src/bbl/powerplay_model.py), [`negbin.py`](src/bbl/negbin.py))

- Runs in the first six overs are overdispersed (variance is about 3x the mean), so a Poisson is out. The model is a **negative binomial (NB2) regression** with innings order, ridge-shrunk venue effects and optional team batting and bowling effects, fitted with analytic gradients.
- Time decay is chosen by half-life; models are refit each season on earlier seasons only, as in deployment.
- The earlier team-level answer is a 50/50 mixture of an all-history fit and a one-year half-life fit, which hedges uncertainty about the league's scoring level ([`powerplay_predict.py`](src/bbl/powerplay_predict.py)).
- **Primary answer, the player layer** ([`player_layer.py`](src/bbl/player_layer.py), [`player_scenarios.py`](src/bbl/player_scenarios.py)): each batter and bowler gets an empirical-Bayes-shrunk powerplay runs-per-ball rating from earlier seasons only (prior strength 60 balls). The line-up's average rating enters as a covariate on the venue-and-innings model. Scenario line-ups for 1 Jan are averaged.

## Evaluation

### Match result (out of sample, 469 matches; coin flip = 0.6931 log loss)

| Model | Log loss | Brier | vs coin flip, 95% bootstrap CI |
|---|---|---|---|
| Home-only | 0.6907 | 0.2488 | -0.0024 [-0.0067, +0.0019] |
| Elo (live) + home | 0.6859 | 0.2464 | -0.0072 [-0.0202, +0.0059] |
| Bradley-Terry | 0.6857 | 0.2461 | -0.0075 [-0.0247, +0.0116] |
| **Ensemble** | **0.6844** | **0.2456** | **-0.0087 [-0.0246, +0.0076]** |

The ensemble beats a coin flip, but **every interval includes zero**. That is the realistic outcome for T20: team strength explains little of any single match. Calibration is noisy (about 117 matches per quartile): the highest predicted quartile averaged 60% and won 52%, and the lowest averaged 40% and won 36%. The gain is concentrated early in each season, which is closest to this fixture: over the first 40% of each season the ensemble's log loss is 0.6689 against 0.6931 for a coin flip. Full table: [`results/match_eval.csv`](results/match_eval.csv).

### Powerplay runs (out of sample, 950 innings, mean log score per innings)

| Model | Log score | vs league-wide NB, 95% CI |
|---|---|---|
| League-wide NB | -3.8866 | n/a |
| NB + venue + innings | -3.8765 | +0.0101 [+0.0000, +0.0204] |
| NB + teams + venue + innings | -3.8776 | +0.0091 [-0.0045, +0.0221] |
| + batter rating (player layer) | -3.8569 | **+0.0196 vs venue+innings** [+0.0058, +0.0340] |
| + batter and bowler ratings | -3.8535 | +0.0230 [+0.0075, +0.0385] |

Calibration is good: the central 80% and 90% intervals covered 81.2% and 89.8% of innings, and the PIT histogram is close to flat. The batter rating is the one component with a clear gain; team effects are not distinguishable from noise.

## Limitations

- **Small signal.** Match-result gains over a coin flip are not statistically established, and the powerplay gains are small in absolute terms.
- **The player-layer backtest is optimistic.** It uses the batters who actually faced powerplay balls (the "XI known" case); pre-toss, personnel are scenarios.
- **Roster churn is only partly modelled.** The ratings predate the 2026/27 squads. The 0.43 headline reflects a judgement-based shrink of the rating gap (keeping 100% of it gives 0.41, 50% gives 0.49; see `match_prediction.json`), which has not been backtested.
- **Scoring level drift.** The last two seasons ran about 4 runs hotter than the models' pre-season means league-wide, while the MCG itself ran low. Whether that persists for a 4pm January start is untested (`powerplay_drift.py`).
- **Unmodelled:** the toss, afternoon conditions, weather, pitch, and players with no BBL history (newcomers get a neutral rating).
- **Distribution width.** The squad-adjusted distribution is a single NB around the scenario-average mean, so it does not carry lineup or level uncertainty beyond the NB dispersion; it is likely a touch too narrow.
- The Renegades' team effects were learned mostly at Marvel Stadium, not the MCG.

## Next steps

- Re-run with the announced XIs and toss on match day (swap the scenario dictionaries for the real lineups).
- Backtest the roster-churn shrink on past season boundaries instead of assuming it.
- Add a wickets-aware ball-by-ball simulation as an alternative powerplay model and compare on log score.
- Ensemble with market prices once available, as a benchmark rather than an input.

## Repository layout

```
src/bbl/
  config.py             paths and the fixture (teams, venue, date): edit here to re-target
  load_cricsheet.py     Cricsheet JSON (zip or folder) -> tidy CSVs
  common.py             shared helpers: filtering, home flags, log loss
  eda.py                exploratory summary and a simple Elo baseline
  match_model.py        Elo + Bradley-Terry backtest (nested selection)
  match_predict.py      deployed win probability, bootstrap, churn sensitivity
  negbin.py             NB2 likelihood with gradient (unit-tested)
  powerplay_model.py    NB regression backtest, calibration
  powerplay_drift.py    is the league scoring level persistent?
  powerplay_predict.py  team-level distributions for the fixture
  player_layer.py       does player-level information add out-of-sample signal?
  player_scenarios.py   squad-adjusted distributions (primary answer)
tests/                  loader, helpers and NB-likelihood tests
results/                committed outputs: CSVs, charts, logs/
data/                   not committed; see data/README.md
```

Run order and dependencies: stages import the stage they build on (`match_predict` imports `match_model`; `powerplay_*` and `player_*` import `powerplay_model`), and that import silently re-runs its backtest. This keeps each result traceable but means a stage costs a few seconds.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
make install      # pip install -e ".[dev]"
make data         # download Cricsheet and build data/*.csv
make all          # eda + match + powerplay + players; logs in results/logs/
make test         # pytest
```

Individual stages: `make eda | match | powerplay | players`. Console output of every run is saved in `results/logs/`.

## Data and credits

Ball-by-ball data from [Cricsheet](https://cricsheet.org). Squad information was gathered from public news (club and cricket.com.au announcements) as at 21 November 2026 and will have changed by match day.
