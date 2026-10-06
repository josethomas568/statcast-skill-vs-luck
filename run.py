"""Skill vs luck in MLB: end-to-end pipeline.

    python run.py                                 # real Statcast, 2023-2026
    python run.py --years 2021 2022 2023 2024 2025 2026
    python run.py --synthetic                     # offline dry run with known true skills

Outputs go to ./output/: charts, CSVs, and summary.html (the one-page coach summary).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from model import METRICS, fit_season          # noqa: E402
from validate import next_season_table, pooled, score  # noqa: E402
import report                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026],
                    help="Consecutive seasons. 2023+ avoids the pitch-clock / shift-ban break.")
    ap.add_argument("--synthetic", action="store_true", help="Use simulated data (no download).")
    ap.add_argument("--refresh", action="store_true", help="Re-download Statcast even if cached.")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    years = sorted(args.years)
    out = Path(args.out); out.mkdir(exist_ok=True)

    truth = None
    if args.synthetic:
        from simulate import simulate
        pa, truth = simulate(seasons=tuple(years))
        names = {}
    else:
        from data import load_statcast_pa, lookup_names
        pa = load_statcast_pa(years, refresh=args.refresh)
        names = {}
    print(f"{len(pa):,} plate appearances across {pa.season.nunique()} seasons")

    # 1. Fit every metric x season
    fits = {}
    for season in years:
        for m in METRICS:
            t = time.time()
            fits[(m, season)] = f = fit_season(pa, m, season)
            print(f"  {season} {METRICS[m][0]:<6} half-signal: hitters {f.half_signal('batter'):6.0f}  "
                  f"pitchers {f.half_signal('pitcher'):6.0f}  park SD {np.sqrt(f.tau2['park']):.4f}  "
                  f"({time.time() - t:.1f}s)")

    players = pd.concat([f.players for f in fits.values()], ignore_index=True)
    parks = pd.concat([f.parks for f in fits.values()], ignore_index=True)
    if not args.synthetic:
        names = lookup_names(players["id"].unique())
    players["name"] = players["id"].map(names)

    # 2. Shrinkage / stabilization
    stab = report.stabilization_table(fits)

    # 3. Next-season validation
    tables = []
    for season in years[:-1]:
        for m in METRICS:
            tb = next_season_table(fits[(m, season)], pa)
            if m == "woba":  # cross-check: does xwOBA skill forecast next-year wOBA better?
                xw = fits[("xwoba", season)].players[["role", "id", "skill"]].rename(columns={"skill": "xwoba_model"})
                tb = tb.merge(xw, on=["role", "id"], how="left")
            tables.append(tb)
    val = pd.concat(tables, ignore_index=True)
    sc = score(val)
    pool = pooled(sc)

    # 4. Outputs
    players.to_csv(out / "skill_estimates.csv", index=False)
    parks.to_csv(out / "park_effects.csv", index=False)
    stab.to_csv(out / "stabilization.csv", index=False)
    sc.to_csv(out / "validation_by_season.csv", index=False)
    pool.to_csv(out / "validation_pooled.csv", index=False)

    latest = {m: fits[(m, years[-1])] for m in METRICS}
    report.fig_reliability(stab, out / "fig_reliability.png")
    report.fig_shrinkage(latest["woba"], out / "fig_shrinkage_woba_hitters.png", "batter")
    report.fig_shrinkage(latest["babip"], out / "fig_shrinkage_babip_pitchers.png", "pitcher")
    report.fig_validation(pool, out / "fig_validation.png")
    report.coach_summary(stab, pool, latest, names, out / "summary.html", out / "summary.md",
                         synthetic=args.synthetic, years=years)

    if truth is not None:  # synthetic only: how close does each method get to the TRUE skill?
        rows = []
        for (m, season), f in fits.items():
            P = f.players.merge(truth[truth.season == season][["role", "id", m]], on=["role", "id"])
            P = P[P.n >= 50]
            for role, d in P.groupby("role"):
                rows.append({"metric": m, "season": season, "role": role,
                             "rmse_raw_vs_truth": np.sqrt(np.average((d.raw - d[m]) ** 2, weights=d.n)),
                             "rmse_model_vs_truth": np.sqrt(np.average((d.skill - d[m]) ** 2, weights=d.n))})
        rec = pd.DataFrame(rows).groupby(["metric", "role"]).mean(numeric_only=True).drop(columns="season")
        rec.to_csv(out / "synthetic_truth_recovery.csv")
        print("\nRecovery of TRUE skill (synthetic):\n", rec.round(4).to_string())

    show = pool[pool.bucket == "all"][["metric", "role", "rmse_raw", "rmse_regressed", "rmse_model",
                                       "gain_vs_raw", "gain_lo", "gain_hi", "players"]]
    print("\nNext-season validation (pooled):\n", show.round(4).to_string(index=False))
    print(f"\nWrote outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
