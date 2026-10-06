"""Charts and the one-page coach / scout summary."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import METRICS, Fit

# Reference palette (validated categorical slots 1-2) and neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
ROLE_COLOR = {"batter": BLUE, "pitcher": ORANGE}
ROLE_LABEL = {"batter": "Hitters", "pitcher": "Pitchers"}
FULL_SEASON = {"batter": 600, "pitcher": 700}   # typical regular's PA / batters faced
BIP_PER_PA = 0.66                               # rough share of PAs ending in a ball in play

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def _pa_equiv(metric: str, n: float) -> float:
    return n / BIP_PER_PA if METRICS[metric][3] == "BIP" else n


def stabilization_table(fits: dict[tuple[str, int], Fit]) -> pd.DataFrame:
    """Half-signal point per metric and role, averaged across seasons (in PA)."""
    rows = []
    for (metric, season), f in fits.items():
        for role in ("batter", "pitcher"):
            hs = _pa_equiv(metric, f.half_signal(role))
            rows.append({"metric": metric, "role": role, "season": season, "half_pa": hs,
                         "skill_sd": np.sqrt(f.tau2[role]), "park_sd": np.sqrt(f.tau2["park"])})
    t = pd.DataFrame(rows).groupby(["metric", "role"]).agg(
        half_pa=("half_pa", "median"), skill_sd=("skill_sd", "median"), park_sd=("park_sd", "median"),
        seasons=("season", "nunique")).reset_index()
    t["full_season_reliability"] = [FULL_SEASON[r] / (FULL_SEASON[r] + h) for r, h in zip(t["role"], t["half_pa"])]
    return t


def fig_reliability(stab: pd.DataFrame, out: Path):
    metrics = list(METRICS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.2), sharey=True)
    n = np.linspace(0, 800, 400)
    for ax, m in zip(axes, metrics):
        for role in ("batter", "pitcher"):
            h = stab[(stab.metric == m) & (stab.role == role)]["half_pa"].iloc[0]
            ax.plot(n, n / (n + h), color=ROLE_COLOR[role], lw=2, label=ROLE_LABEL[role])
            if h <= 800:
                ax.plot([h], [0.5], "o", ms=7, color=ROLE_COLOR[role], mec="white", mew=2, zorder=3)
                ax.annotate(f"{h:.0f}", (h, 0.5), xytext=(6, -15 if role == "batter" else 7),
                            textcoords="offset points", fontsize=8.5, color=INK2)
            else:
                yv = 800 / (800 + h)
                ax.annotate(f"50/50 at ~{h:,.0f}", (800, yv), xytext=(-4, 6 if role == "batter" else -14),
                            textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(METRICS[m][0], color=INK, fontsize=11, loc="left")
        ax.set_xlim(0, 800); ax.set_ylim(0, 1)
        ax.set_xlabel("Plate appearances" if METRICS[m][3] == "PA" else "PA (BABIP: via balls in play)", fontsize=8.5)
        ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Share of the stat that is skill")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("How fast each stat becomes trustworthy (dot = 50/50 skill vs luck)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_shrinkage(fit: Fit, out: Path, role: str = "batter"):
    P = fit.players[(fit.players.role == role) & (fit.players.n >= 20)].copy()
    lab = METRICS[fit.metric][0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.vlines(P["n"], P["raw"], P["skill"], color=MUTED, lw=0.6, alpha=0.6)
    ax.scatter(P["n"], P["raw"], s=12, facecolors="none", edgecolors=MUTED, lw=0.8, label=f"Raw {lab}")
    ax.scatter(P["n"], P["skill"], s=12, color=ROLE_COLOR[role], edgecolors="white", lw=0.4,
               label=f"Skill estimate", zorder=3)
    ax.axhline(fit.mu, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.annotate("average player", (1, fit.mu), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK2,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    lo, hi = np.percentile(P["raw"], [1, 99])
    pad = (hi - lo) * 0.15
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Plate appearances this season" if METRICS[fit.metric][3] == "PA" else "Balls in play this season")
    ax.set_ylabel(lab)
    fig.suptitle(f"{ROLE_LABEL[role]}, {fit.season}: small samples get pulled hard toward average",
                 x=0.01, ha="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9,
              borderaxespad=0.3)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_validation(pool: pd.DataFrame, out: Path):
    d = pool[pool.bucket == "all"].copy()
    d["regressed_gain"] = 1 - d["rmse_regressed"] / d["rmse_raw"]
    order = [(m, r) for m in METRICS for r in ("batter", "pitcher")]
    d = d.set_index(["metric", "role"])
    d = d.loc[[o for o in order if o in d.index]]
    labels = [f"{METRICS[m][0]} · {ROLE_LABEL[r]}" for m, r in d.index]
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.2))
    h = 0.36
    ax.barh(y + h / 2, d["regressed_gain"] * 100, height=h - 0.04, color="#86b6ef",
            label="Simple regression to the mean")
    ax.barh(y - h / 2, d["gain_vs_raw"] * 100, height=h - 0.04, color=BLUE, label="Mixed model (skill estimate)")
    for yi, g in zip(y, d["gain_vs_raw"]):
        ax.annotate(f"{g*100:.0f}%", (g * 100, yi - h / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Cut in next-season forecast error vs. using last season's raw stat")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08 - 0.6 / len(d)), ncol=2, fontsize=8.5)
    ax.set_title("Next-season test: how much better than raw stats?", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- one-page summary

def _verdict(rel: float) -> str:
    if rel >= 0.75:
        return "Mostly skill after a full season"
    if rel >= 0.5:
        return "Part skill, part luck, even over a season"
    return "Mostly luck in a single season"


def _fmt(metric: str, x: float) -> str:
    return f"{x*100:.1f}%" if metric in ("k", "bb") else f"{x:.3f}".lstrip("0")


def coach_summary(stab, pool, latest_fits, names, out_html: Path, out_md: Path, synthetic: bool, years):
    season = latest_fits["woba"].season
    first, last = min(years), max(years)

    # 1. Trust table
    trust = []
    for m in ("k", "bb", "xwoba", "woba", "babip"):
        row = {"stat": METRICS[m][0]}
        for r in ("batter", "pitcher"):
            s = stab[(stab.metric == m) & (stab.role == r)].iloc[0]
            row[r] = s["half_pa"]
            row[r + "_rel"] = s["full_season_reliability"]
            row[r + "_verdict"] = _verdict(s["full_season_reliability"])
        trust.append(row)

    # 2. Validation headline
    allb = pool[pool.bucket == "all"].set_index(["metric", "role"])
    small = pool[pool.bucket == "50-149"].set_index(["metric", "role"])
    g = lambda m, r, t=allb: t.loc[(m, r), "gain_vs_raw"] * 100 if (m, r) in t.index else float("nan")
    xw_note = ""
    if "rmse_xwoba_model" in allb.columns and ("woba", "batter") in allb.index:
        a = allb.loc[("woba", "batter")]
        if pd.notna(a.get("rmse_xwoba_model")):
            better = a["rmse_xwoba_model"] < a["rmse_model"]
            xw_note = ("For hitters, this year's <b>contact-quality (xwOBA)</b> skill predicted next year's wOBA "
                       + ("<b>better</b>" if better else "<b>no better</b>")
                       + " than this year's wOBA skill did.")

    # 3. Worked example + regression watch list (hitters, wOBA, latest season)
    P = latest_fits["woba"].players
    B = P[(P.role == "batter")].copy()
    B["gap"] = B["raw"] - B["skill"]
    B["name"] = B["id"].map(lambda i: names.get(int(i), f"Player {i}"))
    ex_pool = B[(B.n.between(120, 220))]
    ex = ex_pool.loc[ex_pool["raw"].idxmax()] if len(ex_pool) else B.loc[B["raw"].idxmax()]
    reg = B[B.n >= 250]
    over = reg.nlargest(3, "gap")
    under = reg.nsmallest(3, "gap")
    mu = latest_fits["woba"].mu

    def plist(df):
        return "".join(f"<li>{html.escape(r.name)}: {_fmt('woba', r.raw)} actual, "
                       f"{_fmt('woba', r.skill)} skill ({int(r.n)} PA)</li>" for r in df.itertuples())

    banner = ("<div class='banner'>DRY RUN ON SYNTHETIC DATA. These numbers are from simulated "
              "players to test the pipeline. Run <code>python run.py</code> to fill this page "
              "with real Statcast results.</div>") if synthetic else ""

    trust_rows = "".join(
        f"<tr><td>{t['stat']}</td><td>{t['batter']:.0f}</td><td>{t['batter_verdict']}</td>"
        f"<td>{t['pitcher']:.0f}</td><td>{t['pitcher_verdict']}</td></tr>" for t in trust)

    T = {t["stat"]: t for t in trust}
    k_note = ", smaller because K% is mostly skill to begin with" if T["K%"]["batter_rel"] >= 0.75 else ""
    tips = [f"<li><b>Strikeouts settle fastest</b>: about {T['K%']['batter']:.0f} PA for hitters and "
            f"{T['K%']['pitcher']:.0f} batters faced for pitchers to be half skill. React to K% early.</li>"]
    if T["BABIP"]["pitcher_rel"] < 0.5:
        tips.append("<li><b>Don't trust BABIP</b>, especially for pitchers: even a full season of hits on "
                    "balls in play is mostly luck and defense.</li>")
    if T["xwOBA"]["batter"] < T["wOBA"]["batter"]:
        tips.append(f"<li><b>Prefer contact quality (xwOBA) to results (wOBA)</b> for short stretches: it is half "
                    f"skill after ~{T['xwOBA']['batter']:.0f} PA versus ~{T['wOBA']['batter']:.0f} for wOBA.</li>")
    tips.append(f"<li>Always ask \"how many PA?\" A hitter's wOBA is not even half skill until about "
                f"{T['wOBA']['batter']:.0f} PA; for pitchers it takes about {T['wOBA']['pitcher']:.0f}.</li>")
    advice = "".join(tips)

    css = """
    @page { size: letter; margin: 0.5in; }
    body { font: 10.5px/1.42 -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color:#0b0b0b;
           max-width: 7.5in; margin: 0 auto; padding: 16px; background:#fff; }
    h1 { font-size: 19px; margin: 0 0 2px; } h2 { font-size: 12.5px; margin: 12px 0 4px; color:#1c5cab; }
    .sub { color:#52514e; margin: 0 0 8px; } p, ul { margin: 3px 0; } ul { padding-left: 17px; }
    table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 10px; }
    th, td { text-align: left; padding: 3px 6px; border-bottom: 1px solid #e6e5e1; vertical-align: top; }
    th { color:#52514e; font-weight: 600; } td:nth-child(2), td:nth-child(4) { font-variant-numeric: tabular-nums; }
    .box { background:#f4f8fe; border-left: 3px solid #2a78d6; padding: 6px 10px; margin: 6px 0; }
    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .banner { background:#fff4e0; border:1px solid #eda100; padding:5px 8px; margin-bottom:8px; font-weight:600; }
    .foot { color:#8a8984; font-size: 9px; margin-top: 10px; }
    """
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skill or Luck?</title><style>{css}</style></head><body>
{banner}
<h1>Skill or Luck? Reading a player's numbers the right way</h1>
<p class="sub">MLB Statcast, {first}–{last} regular seasons · every plate appearance, adjusted for park and opponent</p>

<h2>The idea</h2>
<p>Every stat line is <b>skill + context + luck</b>. Context is the park and the quality of the opponents. Luck is
which balls found holes. Our model estimates all three at once and keeps only the skill. The smaller the sample, the
more of a hot or cold start is luck, so the more we pull the number back toward average.</p>

<h2>How long before you can trust a stat?</h2>
<p>The number is the plate appearances (batters faced, for pitchers) at which a stat is <b>half skill, half luck</b>.
Lower means it settles faster.</p>
<table><tr><th>Stat</th><th>Hitters</th><th>After a full season</th><th>Pitchers</th><th>After a full season</th></tr>
{trust_rows}</table>

<div class="box"><b>Rule of thumb:</b> Skill = average + (share that is skill) × (player − average), with
share = PA ÷ (PA + the number in the table). <br>Example: {html.escape(ex['name'])} hit
{_fmt('woba', ex.raw)} wOBA in {int(ex.n)} PA in {season}. Average is {_fmt('woba', mu)}. Only about
{ex.reliability*100:.0f}% of his edge is likely skill, so his estimate is <b>{_fmt('woba', ex.skill)}</b>.</div>

<div class="cols"><div>
<h2>Did it work? Tested on the next season</h2>
<p>We made estimates from one season and checked them against what players did the following year
({first}→{first+1} through {last-1}→{last}).</p>
<ul>
<li>Hitters' wOBA: <b>{g('woba','batter'):.0f}% less error</b> than using last year's raw number.</li>
<li>Pitchers' wOBA allowed: <b>{g('woba','pitcher'):.0f}% less error</b>.</li>
<li>On small samples (50–149 PA): hitters {g('woba','batter',small):.0f}%, pitchers {g('woba','pitcher',small):.0f}%.</li>
<li>Strikeout rate: {g('k','batter'):.0f}% (hitters), {g('k','pitcher'):.0f}% (pitchers){k_note}.</li>
</ul>
<p>{xw_note}</p>
</div><div>
<h2>{season} hitters likely to regress</h2>
<p><b>Running hot</b> (results ahead of skill):</p><ul>{plist(over)}</ul>
<p><b>Running cold</b> (better than their results):</p><ul>{plist(under)}</ul>
</div></div>

<h2>How to use this</h2>
<ul>{advice}</ul>
<p class="foot">Method: linear mixed model per stat and season with crossed random effects for hitter, pitcher, and park,
plus a same-handed matchup adjustment; variances by REML; next-season test uses players with ≥50 trials in season one
and ≥100 in season two, error weighted by season-two playing time. Defense, catcher framing, and aging are not modeled.</p>
</body></html>"""
    out_html.write_text(page, encoding="utf-8")

    md = [f"# Skill or Luck? ({first}–{last})", "",
          "| Stat | Hitters: PA to 50/50 | Pitchers: BF to 50/50 |", "|---|---|---|"]
    md += [f"| {t['stat']} | {t['batter']:.0f} | {t['pitcher']:.0f} |" for t in trust]
    md += ["", f"Next-season wOBA error vs raw: hitters -{g('woba','batter'):.0f}%, pitchers -{g('woba','pitcher'):.0f}%.",
           "", "See summary.html for the full one-page version."]
    out_md.write_text("\n".join(md), encoding="utf-8")
