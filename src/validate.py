"""Next-season validation: do skill estimates from season Y predict season Y+1
better than season Y's raw stats?

Four forecasts for each player's Y+1 rate, all built only from season Y:
  raw        - his season-Y rate as-is
  league     - league average (the "know nothing" floor)
  regressed  - classic regression to the mean: (n*raw + k*avg) / (n + k), no context
  model      - the mixed-model skill estimate (shrunk AND adjusted for park/opponent)

Scored by RMSE against the Y+1 observed rate, weighted by Y+1 sample size,
for players with enough trials in both seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import METRICS, Fit

FORECASTS = ("raw", "league", "regressed", "model")
BUCKETS = [(50, 150, "50-149"), (150, 400, "150-399"), (400, 10_000, "400+")]


def next_season_table(fit: Fit, pa: pd.DataFrame, min_y: int = 50, min_next: int = 100) -> pd.DataFrame:
    _, ycol, filt, _ = METRICS[fit.metric]
    nxt = pa[pa["season"] == fit.season + 1]
    if filt:
        nxt = nxt.query(filt)
    cur = pa[pa["season"] == fit.season]
    if filt:
        cur = cur.query(filt)
    league = cur[ycol].mean()

    out = []
    for role in ("batter", "pitcher"):
        tgt = nxt.groupby(role)[ycol].agg(["mean", "size"]).rename(columns={"mean": "next", "size": "n_next"})
        P = fit.players[fit.players["role"] == role].set_index("id")
        d = P.join(tgt, how="inner")
        d = d[(d["n"] >= min_y) & (d["n_next"] >= min_next)].copy()
        k = fit.half_signal(role)
        d["league"] = league
        d["regressed"] = (d["n"] * d["raw"] + k * league) / (d["n"] + k)
        d["model"] = d["skill"]
        out.append(d.reset_index().rename(columns={"index": "id"}))
    return pd.concat(out, ignore_index=True)


def _wrmse(pred, target, w):
    return float(np.sqrt(np.average((pred - target) ** 2, weights=w)))


def score(table: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """RMSE of each forecast, overall and by season-Y sample size, plus a bootstrap
    interval on the model's improvement over raw stats."""
    rng = np.random.default_rng(seed)
    rows = []
    for (metric, season, role), d in table.groupby(["metric", "season", "role"]):
        groups = [("all", d)] + [(lab, d[(d["n"] >= lo) & (d["n"] < hi)]) for lo, hi, lab in BUCKETS]
        for lab, g in groups:
            if len(g) < 15:
                continue
            r = {"metric": metric, "season": season, "role": role, "bucket": lab, "players": len(g)}
            for f in FORECASTS + ("xwoba_model",):
                if f in g.columns and g[f].notna().all():
                    r[f"rmse_{f}"] = _wrmse(g[f], g["next"], g["n_next"])
            r["corr_raw"] = float(np.corrcoef(g["raw"], g["next"])[0, 1])
            r["corr_model"] = float(np.corrcoef(g["model"], g["next"])[0, 1])
            r["gain_vs_raw"] = 1 - r["rmse_model"] / r["rmse_raw"]
            if lab == "all":
                idx = np.arange(len(g))
                gains = []
                for _ in range(n_boot):
                    s = g.iloc[rng.choice(idx, len(idx))]
                    gains.append(1 - _wrmse(s["model"], s["next"], s["n_next"])
                                 / _wrmse(s["raw"], s["next"], s["n_next"]))
                r["gain_lo"], r["gain_hi"] = np.percentile(gains, [5, 95])
            rows.append(r)
    return pd.DataFrame(rows)


def pooled(scores: pd.DataFrame) -> pd.DataFrame:
    """Average across season pairs (player-count weighted) for the headline table."""
    cols = [c for c in scores.columns if c.startswith(("rmse_", "corr_", "gain"))]
    rows = []
    for (metric, role, bucket), g in scores.groupby(["metric", "role", "bucket"]):
        r = {"metric": metric, "role": role, "bucket": bucket,
             "players": int(g["players"].sum()), "pairs": len(g)}
        for c in cols:
            ok = g[c].notna()
            r[c] = float(np.average(g.loc[ok, c], weights=g.loc[ok, "players"])) if ok.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)
