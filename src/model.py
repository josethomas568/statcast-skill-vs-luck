"""Crossed random-effects model separating player skill from context and noise.

For one season and one outcome, each plate appearance i is modelled as

    y_i = mu + beta * same_hand_i + batter[b(i)] + pitcher[p(i)] + park[k(i)] + e_i

    batter ~ N(0, tau_b^2)   pitcher ~ N(0, tau_p^2)   park ~ N(0, tau_k^2)   e ~ N(0, sigma^2)

Every hitter's estimate is adjusted for the pitchers he faced and the parks he
played in, and every pitcher's for the hitters and parks he faced, all at once.
The variance components are estimated by restricted maximum likelihood (REML,
i.e. empirical Bayes); the player estimates are the posterior means (BLUPs), which are
automatically shrunk toward average by the right amount for their sample size.

This is the Gaussian (linear-probability) version: exact, fast, and transparent.
With rates in the 0.05 to 0.35 range and hundreds of PAs, it agrees closely with
a logistic hierarchical model; see README for how to swap one in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

METRICS = {
    # key: (label, outcome column, row filter, denominator label)
    "k":     ("K%",    "k",       None,       "PA"),
    "bb":    ("BB%",   "bb",      None,       "PA"),
    "woba":  ("wOBA",  "woba",    None,       "PA"),
    "xwoba": ("xwOBA", "xwoba",   None,       "PA"),
    "babip": ("BABIP", "hit_bip", "bip == 1", "BIP"),
}
GROUPS = ("batter", "pitcher", "park")


@dataclass
class Fit:
    metric: str
    season: int
    mu: float                 # league mean at the average platoon mix
    beta_same_hand: float
    sigma2: float             # PA-level noise variance
    tau2: dict                # between-player (or park) variance of true skill
    players: pd.DataFrame     # per batter / pitcher estimates
    parks: pd.DataFrame
    iterations: int

    def half_signal(self, role: str) -> float:
        """Sample size at which an observed rate is 50% skill, 50% noise (= sigma^2 / tau^2)."""
        return self.sigma2 / self.tau2[role]

    def reliability(self, role: str, n) -> np.ndarray:
        """Share of the observed deviation from average that is real skill, after n trials."""
        n = np.asarray(n, dtype=float)
        return n / (n + self.half_signal(role))


def fit_season(pa: pd.DataFrame, metric: str, season: int, max_iter: int = 500, tol: float = 1e-6) -> Fit:
    label, ycol, filt, _ = METRICS[metric]
    d = pa[pa["season"] == season]
    if filt:
        d = d.query(filt)
    y = d[ycol].to_numpy(float)
    n = len(y)

    codes, levels = {}, {}
    for g in GROUPS:
        c, lv = pd.factorize(d[g], sort=True)
        codes[g], levels[g] = c, lv
    q = {g: len(levels[g]) for g in GROUPS}

    # Design: [intercept, same_hand | batter one-hot | pitcher one-hot | park one-hot]
    sh = d["same_hand"].to_numpy(float)
    blocks = [sp.csr_matrix(np.column_stack([np.ones(n), sh - sh.mean()]))]
    for g in GROUPS:
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), codes[g])), shape=(n, q[g])))
    Z = sp.hstack(blocks).tocsr()
    A = (Z.T @ Z).toarray()
    b = Z.T @ y
    p = A.shape[0]

    sl, start = {}, 2
    for g in GROUPS:
        sl[g] = slice(start, start + q[g])
        start += q[g]

    yy = y @ y
    p_fixed = 2
    diag_idx = np.arange(p)
    A_diag = A[diag_idx, diag_idx].copy()

    def neg2_reml(log_lam):
        """Profiled REML criterion in the noise-to-signal ratios lambda_g = sigma^2 / tau_g^2."""
        lam = np.zeros(p)
        for g, ll in zip(GROUPS, log_lam):
            lam[sl[g]] = np.exp(ll)
        A[diag_idx, diag_idx] = A_diag + lam
        try:
            cf = cho_factor(A, lower=True, check_finite=False)
        finally:
            A[diag_idx, diag_idx] = A_diag
        theta = cho_solve(cf, b, check_finite=False)
        pen_rss = yy - b @ theta
        logdet = 2 * np.log(np.diag(cf[0])).sum()
        return ((n - p_fixed) * np.log(pen_rss / (n - p_fixed)) + logdet
                - sum(q[g] * ll for g, ll in zip(GROUPS, log_lam)))

    # Start from method-of-moments ratios: between-player variance of raw rates
    # minus the binomial-style sampling variance.
    s2 = y.var()
    start_ll = []
    for g in GROUPS:
        cnt = np.bincount(codes[g], minlength=q[g])
        mean = np.bincount(codes[g], weights=y, minlength=q[g]) / np.maximum(cnt, 1)
        ok = cnt >= 50
        t2 = np.average((mean[ok] - y.mean()) ** 2, weights=cnt[ok]) - s2 * ok.sum() / cnt[ok].sum()
        start_ll.append(np.log(s2 / max(t2, s2 / 5000)))
    res = minimize(neg2_reml, start_ll, method="L-BFGS-B",
                   bounds=[(np.log(1.0), np.log(1e7))] * len(GROUPS),
                   options={"maxiter": max_iter, "ftol": tol * 1e-3})
    it = res.nit

    lam = np.zeros(p)
    ratios = dict(zip(GROUPS, np.exp(res.x)))
    for g in GROUPS:
        lam[sl[g]] = ratios[g]
    cf = cho_factor(A + np.diag(lam), lower=True, check_finite=False)
    theta = cho_solve(cf, b, check_finite=False)
    sigma2 = (yy - b @ theta) / (n - p_fixed)
    tau2 = {g: sigma2 / ratios[g] for g in GROUPS}

    # Final solve at the REML variance components
    cf = cho_factor(A + np.diag(lam), lower=True, check_finite=False)
    theta = cho_solve(cf, b, check_finite=False)
    C = cho_solve(cf, np.eye(p), check_finite=False)
    post_var = sigma2 * np.diag(C)
    mu = theta[0]

    rows = []
    for g in ("batter", "pitcher"):
        cnt = np.bincount(codes[g], minlength=q[g])
        raw = np.bincount(codes[g], weights=y, minlength=q[g]) / np.maximum(cnt, 1)
        u, v = theta[sl[g]], post_var[sl[g]]
        rows.append(pd.DataFrame({
            "metric": metric, "season": season, "role": g, "id": levels[g].astype(int),
            "n": cnt, "raw": raw,
            "skill": mu + u,                         # context-neutral estimate
            "skill_sd": np.sqrt(v + post_var[0]),
            "reliability": 1 - v / tau2[g],          # how much of the raw deviation we keep
        }))
    players = pd.concat(rows, ignore_index=True)

    pk_cnt = np.bincount(codes["park"], minlength=q["park"])
    parks = pd.DataFrame({
        "metric": metric, "season": season, "park": levels["park"],
        "n": pk_cnt, "effect": theta[sl["park"]], "effect_sd": np.sqrt(post_var[sl["park"]]),
    })
    return Fit(metric, season, mu, theta[1], sigma2, tau2, players, parks, it)
