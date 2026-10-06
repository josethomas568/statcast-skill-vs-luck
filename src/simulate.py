"""Synthetic Statcast-shaped plate appearances with KNOWN true skills.

Used to test the pipeline without network access, and to check that the model
recovers true skill better than raw stats do. Output has the same columns as
data.pitches_to_pa(), plus a truth table of each player's real skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W = {"bb": 0.69, "1b": 0.88, "2b": 1.25, "3b": 1.58, "hr": 2.03}  # approximate wOBA weights
HIT_MIX = np.array([0.70, 0.26, 0.04])                         # 1B / 2B / 3B share of BIP hits
HIT_W = np.array([W["1b"], W["2b"], W["3b"]])

BASE = {"k": 0.225, "bb": 0.082, "hr": 0.031}  # league rates per PA
BASE_BABIP = 0.292

# Spread of true talent, on the logit scale.
SD_BAT = {"k": 0.30, "bb": 0.30, "hr": 0.45, "babip": 0.09}
SD_PIT = {"k": 0.28, "bb": 0.25, "hr": 0.15, "babip": 0.04}
SD_PARK = {"k": 0.03, "bb": 0.02, "hr": 0.12, "babip": 0.04}
YEAR_TO_YEAR = 0.88      # persistence of true talent between seasons
SAME_HAND_K = 0.15       # platoon disadvantage, logit points of K


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _rates(kb, bbb, hrb, babb, kp=0, bbp=0, hrp=0, babp=0, kpk=0, bbpk=0, hrpk=0, babpk=0, same=0):
    """Outcome probabilities for a PA given additive logit offsets."""
    p_bip = 1 - BASE["k"] - BASE["bb"] - BASE["hr"]
    lk = np.log(BASE["k"] / p_bip) + kb + kp + kpk + SAME_HAND_K * same
    lb = np.log(BASE["bb"] / p_bip) + bbb + bbp + bbpk
    lh = np.log(BASE["hr"] / p_bip) + hrb + hrp + hrpk
    den = 1 + np.exp(lk) + np.exp(lb) + np.exp(lh)
    pk, pb, ph = np.exp(lk) / den, np.exp(lb) / den, np.exp(lh) / den
    pbip = 1 / den
    babip = _expit(_logit(BASE_BABIP) + babb + babp + babpk)
    woba = pb * W["bb"] + ph * W["hr"] + pbip * babip * (HIT_MIX @ HIT_W)
    return pk, pb, ph, pbip, babip, woba


def simulate(seasons=(2023, 2024, 2025, 2026), n_bat=650, n_pit=820, pa_per_season=184_000, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(30)]

    def new_players(n, sd, start_id):
        sk = {m: rng.normal(0, s, n) for m, s in sd.items()}
        sk["hand"] = rng.random(n) < 0.6     # True = right-handed
        sk["team"] = rng.integers(0, 30, n)
        sk["pt"] = rng.normal(0, 1.1, n)     # playing-time propensity
        sk["id"] = np.arange(start_id, start_id + n)
        return pd.DataFrame(sk)

    bats = new_players(n_bat, SD_BAT, 100000)
    pits = new_players(n_pit, SD_PIT, 500000)
    next_id = {"b": 100000 + n_bat, "p": 500000 + n_pit}
    rows, truth = [], []

    for si, season in enumerate(seasons):
        if si > 0:  # talent drifts, some players leave and are replaced
            for df, sd, key in ((bats, SD_BAT, "b"), (pits, SD_PIT, "p")):
                for m, s in sd.items():
                    df[m] = YEAR_TO_YEAR * df[m] + np.sqrt(1 - YEAR_TO_YEAR**2) * rng.normal(0, s, len(df))
                df["pt"] = 0.8 * df["pt"] + 0.6 * rng.normal(0, 1.0, len(df))
                leave = rng.random(len(df)) < 0.15
                fresh = new_players(int(leave.sum()), sd, next_id[key])
                next_id[key] += int(leave.sum())
                df.loc[leave, fresh.columns] = fresh.values
                for m in list(sd) + ["pt"]:
                    df[m] = df[m].astype(float)
                df["id"] = df["id"].astype(int)
                df["team"] = df["team"].astype(int)
                df["hand"] = df["hand"].astype(bool)
        park = {m: rng.normal(0, s, 30) for m, s in SD_PARK.items()} if si == 0 else park

        # Better hitters (and better pitchers) get more playing time.
        bat_quality = 0.8 * bats["hr"] - 0.6 * bats["k"] + 0.5 * bats["bb"] + 2 * bats["babip"]
        pit_quality = 0.8 * pits["k"] - 0.6 * pits["bb"] - 0.5 * pits["hr"]
        nb = np.round(720 * _expit(1.4 * bats["pt"].values + 0.9 * bat_quality.values - 0.5)).astype(int) + 1
        npit = np.round(780 * _expit(1.4 * pits["pt"].values + 0.9 * pit_quality.values - 1.1)).astype(int) + 1
        bi = rng.permutation(np.repeat(np.arange(len(bats)), nb))
        pi = rng.permutation(np.repeat(np.arange(len(pits)), npit))
        n = min(len(bi), len(pi), pa_per_season)
        bi, pi = bi[:n], pi[:n]
        home_is_bat = rng.random(n) < 0.5
        tm = np.where(home_is_bat, bats["team"].values[bi], pits["team"].values[pi])
        same = (bats["hand"].values[bi] == pits["hand"].values[pi]).astype(int)

        B = {m: bats[m].values[bi] for m in SD_BAT}
        P = {m: pits[m].values[pi] for m in SD_PIT}
        K = {m: park[m][tm] for m in SD_PARK}
        pk, pb, ph, pbip, babip, _ = _rates(B["k"], B["bb"], B["hr"], B["babip"],
                                            P["k"], P["bb"], P["hr"], P["babip"],
                                            K["k"], K["bb"], K["hr"], K["babip"], same)
        u = rng.random(n)
        is_k = u < pk
        is_bb = (u >= pk) & (u < pk + pb)
        is_hr = (u >= pk + pb) & (u < pk + pb + ph)
        is_bip = u >= pk + pb + ph
        hit = is_bip & (rng.random(n) < babip)
        htype = rng.choice(3, n, p=HIT_MIX)
        woba = np.where(is_bb, W["bb"], 0.0) + np.where(is_hr, W["hr"], 0.0) + np.where(hit, HIT_W[htype], 0.0)
        # xwOBA on contact: expected value of the BIP given contact quality, plus
        # tracking noise that is much smaller than the hit / out coin flip.
        xw_bip = babip * (HIT_MIX @ HIT_W) + rng.normal(0, 0.15, n)
        xwoba = np.where(is_bip, np.clip(xw_bip, 0, 2.0), woba)

        rows.append(pd.DataFrame({
            "season": season, "game_pk": season * 100000 + rng.integers(0, 2430, n),
            "batter": bats["id"].values[bi], "pitcher": pits["id"].values[pi],
            "park": np.array(teams)[tm], "same_hand": same,
            "k": is_k.astype(int), "bb": is_bb.astype(int), "woba": woba, "xwoba": xwoba,
            "bip": is_bip.astype(int), "hit_bip": hit.astype(int),
        }))

        # Truth = expected rate vs a league-average opponent in a neutral park,
        # at the league platoon mix.
        for role, df, sd in (("batter", bats, SD_BAT), ("pitcher", pits, SD_PIT)):
            z = {m: df[m].values for m in sd}
            args = (z["k"], z["bb"], z["hr"], z["babip"]) if role == "batter" else \
                (0, 0, 0, 0, z["k"], z["bb"], z["hr"], z["babip"])
            r0 = _rates(*args) if role == "batter" else _rates(*args)
            r1 = _rates(*args, same=1) if role == "batter" else _rates(*args, same=1)
            frac_same = same.mean()
            mix = [(1 - frac_same) * a + frac_same * b for a, b in zip(r0, r1)]
            pk_, pb_, ph_, pbip_, babip_, woba_ = mix
            t = pd.DataFrame({"season": season, "role": role, "id": df["id"].values,
                              "k": pk_, "bb": pb_, "woba": woba_, "babip": babip_,
                              "xwoba": woba_})
            truth.append(t)

    return pd.concat(rows, ignore_index=True), pd.concat(truth, ignore_index=True)
