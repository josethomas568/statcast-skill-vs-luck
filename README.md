# Skill vs. Luck in MLB: a hierarchical model on Statcast

Separates each hitter's and pitcher's true skill from park, opponent, and random variation, measures how much each stat
needs to be shrunk at a given sample size, and checks the estimates the way front offices do: by predicting the next
season.

## Quick start

```bash
pip install -r requirements.txt
python run.py                      # pulls 2023-2026 regular seasons from Baseball Savant, then fits everything
python run.py --synthetic          # offline dry run on simulated players with known true skill
```

The first real run downloads about 3M pitches (roughly 10 to 20 minutes per season) and caches them in `data_cache/`.
Later runs take about 2 minutes. Seasons default to 2023+ because the pitch clock and shift ban changed the run
environment, BABIP especially; pass `--years 2021 2022 ...` to include earlier years.

## What it does

**Data** (`src/data.py`). Pitch-level Statcast reduced to one row per plate appearance in the wOBA universe (no IBB,
sac bunts, or catcher interference). Outcomes: K, BB, wOBA value, xwOBA (Statcast's contact-quality estimate on balls in
play, actual value otherwise), and hit-on-ball-in-play for BABIP.

**Model** (`src/model.py`). For each stat and season, a linear mixed model with *crossed* random effects:

```
y_pa = mu + beta * same_handed + hitter_skill + pitcher_skill + park_effect + noise
```

Every hitter is adjusted for the pitchers he faced and every pitcher for the hitters he faced, simultaneously, plus
park. Variance components come from profiled REML (empirical Bayes). Player estimates are the posterior means (BLUPs)
with standard errors and a per-player reliability, the share of his raw deviation from average the model keeps.
The ~1,500 x 1,500 normal equations are solved densely by Cholesky, so each fit takes a few seconds.

**Shrinkage**. For each stat and role, the *half-signal point* k = sigma^2 / tau^2 is the sample size where a raw
rate is 50% skill. Reliability after n trials is n / (n + k). That gives the coach-friendly rule
`skill = average + n/(n+k) * (raw - average)`.

**Validation** (`src/validate.py`). Fit season Y, forecast each player's Y+1 rate (≥50 trials in Y, ≥100 in Y+1),
RMSE weighted by Y+1 playing time. Compared against: raw Y stat, league average, and classic regression to the mean
with no context adjustment. That last baseline isolates what the park/opponent adjustment adds beyond shrinkage.
Results are broken out by sample-size bucket with bootstrap intervals, plus a cross-check of whether xwOBA skill
predicts next-year wOBA better than wOBA skill does.

## Outputs (`output/`)

| File | What |
|---|---|
| `summary.html` | The one-page coach/scout summary (prints to one letter page) |
| `fig_reliability.png` | Share-of-skill curves by stat, hitters vs pitchers |
| `fig_shrinkage_*.png` | Raw vs. skill estimate by sample size |
| `fig_validation.png` | Next-season error reduction vs. raw stats |
| `skill_estimates.csv` | Every player, stat, season: raw, skill, SE, reliability |
| `park_effects.csv`, `stabilization.csv`, `validation_*.csv` | Supporting tables |

## Dry run (synthetic data)

`output_synthetic/` holds a run on 736k simulated PAs whose true talents are known. The model cut error against
*true* skill by 14% (K%) to 77% (pitcher BABIP) relative to raw stats, for every stat and role
(`synthetic_truth_recovery.csv`). The synthetic numbers only show the code works; the real stabilization points
come from your run. In the simulation, park and opponent are assigned at random, so context adjustment adds almost nothing over
plain regression to the mean. Real schedules are unbalanced (division opponents, home parks), so expect the model's
edge over the "regressed" baseline to be larger on real data. That comparison is the interesting result to look for.

## Limitations and next steps

- **Gaussian on binary outcomes.** Linear-probability mixed models are accurate for rates between ~5% and 35% but
  slightly over-shrink extremes. A logistic version (e.g. `bambi`/PyMC with `(1|batter) + (1|pitcher) + (1|park)`)
  is the natural upgrade; this code is the fast exact baseline to check it against.
- **Shrinkage target.** Everyone shrinks toward the average *player*. Low-playing-time players are usually worse than
  that, so adding log(PA) or prior-season skill as a level-2 predictor should help the small-sample bucket.
- **Not modeled:** fielders behind the pitcher (matters for BABIP), catcher framing, aging, injuries, and multi-year
  pooling (a Marcel-style weighted history would beat a single season).
- **Park proxy** is the home team within each season, which correctly handles the A's and Rays venue changes.
