"""Data layer: pull Statcast pitch-level data with pybaseball and reduce it to one row
per plate appearance (PA), with the outcome columns the model needs.

The PA table has one row per plate appearance and these columns:
    season, game_pk, batter, pitcher, park, same_hand,
    k, bb, woba, xwoba, bip, hit_bip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data_cache"

# Regular-season windows (generous; the game_type filter does the real work).
SEASON_WINDOWS = {y: (f"{y}-03-15", f"{y}-10-05") for y in range(2015, 2031)}

K_EVENTS = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk"}  # intentional walks have woba_denom == 0 and drop out anyway
HIT_EVENTS = {"single", "double", "triple"}


def fetch_season(year: int, refresh: bool = False) -> pd.DataFrame:
    """Download one regular season of pitch-level Statcast data (cached to parquet)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"statcast_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    from pybaseball import cache as pb_cache, statcast

    pb_cache.enable()  # pybaseball's own per-day cache makes interrupted pulls resumable
    start, end = SEASON_WINDOWS[year]
    print(f"Downloading Statcast {year} ({start} to {end}); this takes a while the first time...")
    df = statcast(start_dt=start, end_dt=end, parallel=True)
    keep = [
        "game_type", "game_year", "game_date", "game_pk", "at_bat_number", "pitch_number",
        "batter", "pitcher", "home_team", "stand", "p_throws", "events", "bb_type",
        "woba_value", "woba_denom", "estimated_woba_using_speedangle",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_parquet(path, index=False)
    return df


def pitches_to_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce pitch-level rows to plate appearances with modelling outcomes."""
    df = df[df["game_type"] == "R"]
    pa = df[df["events"].notna() & (df["events"] != "")].copy()
    # wOBA universe: excludes IBB, sacrifice bunts, catcher interference.
    pa = pa[pa["woba_denom"] == 1]
    pa = pa.drop_duplicates(["game_pk", "at_bat_number"])

    ev = pa["events"]
    out = pd.DataFrame({
        "season": pa["game_year"].astype(int).values,
        "game_pk": pa["game_pk"].values,
        "batter": pa["batter"].astype(int).values,
        "pitcher": pa["pitcher"].astype(int).values,
        # Park proxy = home team within a season (handles the A's and Rays moves,
        # because each season is fit separately).
        "park": pa["home_team"].values,
        "same_hand": (pa["stand"] == pa["p_throws"]).astype(int).values,
        "k": ev.isin(K_EVENTS).astype(int).values,
        "bb": ev.isin(BB_EVENTS).astype(int).values,
        "woba": pa["woba_value"].fillna(0).astype(float).values,
    })
    # Ball in play, excluding home runs and sac bunts -> BABIP denominator.
    bip = pa["bb_type"].notna() & ~ev.isin({"home_run", "sac_bunt"})
    out["bip"] = bip.astype(int).values
    out["hit_bip"] = (bip & ev.isin(HIT_EVENTS)).astype(int).values
    # xwOBA per PA: Statcast's contact-quality estimate on balls in play,
    # actual wOBA value otherwise (K, BB, HBP have no luck component on contact).
    xw = pa["estimated_woba_using_speedangle"]
    on_contact = pa["bb_type"].notna() & xw.notna()
    out["xwoba"] = np.where(on_contact.values, xw.values, out["woba"].values).astype(float)
    return out


def load_statcast_pa(years: list[int], refresh: bool = False) -> pd.DataFrame:
    frames = []
    for y in years:
        pa_path = CACHE / f"pa_{y}.parquet"
        if pa_path.exists() and not refresh:
            frames.append(pd.read_parquet(pa_path))
            continue
        pa = pitches_to_pa(fetch_season(y, refresh=refresh))
        pa.to_parquet(pa_path, index=False)
        frames.append(pa)
    return pd.concat(frames, ignore_index=True)


def lookup_names(ids) -> dict[int, str]:
    """MLBAM id -> 'First Last'. Falls back to the id if the lookup is unavailable."""
    ids = sorted({int(i) for i in ids})
    try:
        from pybaseball import playerid_reverse_lookup

        t = playerid_reverse_lookup(ids, key_type="mlbam")
        return {
            int(r.key_mlbam): f"{str(r.name_first).title()} {str(r.name_last).title()}"
            for r in t.itertuples()
        }
    except Exception:
        return {}
