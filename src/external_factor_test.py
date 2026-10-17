"""
Held-out test for external factors: does an outside signal (weather, holidays, conflict
dates) predict Stories' sales on days the model never saw?

An in-sample correlation can appear by chance, especially after looking at many factors.
This script asks a stricter question for each factor:

1. Held-out gain. Fit a transactions-only Poisson baseline (branch x weekday + branch growth)
   on one half of the timeline, and the same model plus the factor. Score both on the other
   half (mean Poisson deviance, lower is better). Run both directions:
   Jan-Mar -> Apr-Jun (the team's primary split, see EVALUATION.md) and Apr-Jun -> Jan-Mar.
2. Placebo. Slide the factor's daily series by every offset of 14-167 days (circular shift,
   which keeps its seasonality and autocorrelation but puts it on the wrong dates), refit and
   rescore. p = share of placebo offsets that do at least as well as the real dates.
   Holm-corrected across all factor/target tests.
3. Positive control. Plant a known effect (e.g. 10% fewer orders on rainy days) into the real
   counts by binomial thinning and check how often step 1 + the placebo threshold catch it.
   This tells us whether "not detected" means "small" or "the test cannot see it".

A factor passes only if it improves held-out deviance in BOTH directions and its Holm-
adjusted placebo p < 0.05. Passing is evidence of predictive association, not causation.

IMPORTANT: weather here is the realised (observed) weather, so this is a retrospective test
under perfect weather knowledge, not an operational forecast. Conflict dates were selected
after the fact. Branch locations are unverified; weather is tested at three candidate
locations (Beirut, Batroun, Broummana). None of these inputs enter the strict model.

Run from the repo root:
    python src/external_factor_test.py            # full run, writes reports/ outputs
    python src/external_factor_test.py --quick    # fewer placebo offsets / simulations

Outputs:
    reports/external_factor_test.csv         one row per factor x target
    reports/external_factor_test.json        same plus protocol, power results, provenance
    reports/figures/external_factor_holdout.png

Requires: pandas, numpy, matplotlib
"""

# %% Setup
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
DATA = ROOT / "data" / "stories_transactions.csv"
EXT = ROOT / "data" / "external"
OUT = ROOT / "reports"
FIG = OUT / "figures"
QUICK = "--quick" in sys.argv

SPLIT = pd.Timestamp("2026-04-01")  # first test date of the primary split (EVALUATION.md)
SHIFTS = list(range(14, 168, 7 if QUICK else 1))
N_SIM = 20 if QUICK else 100
RIDGE = 1e-4
RNG = np.random.default_rng(490)

# Same dates as notebooks/03_holidays.ipynb (Office Holidays list; Islamic dates vary by a day).
HOLIDAYS = ["2026-01-01", "2026-01-06", "2026-02-09", "2026-02-14", "2026-03-20", "2026-03-21",
            "2026-03-25", "2026-04-03", "2026-04-05", "2026-04-10", "2026-04-12", "2026-05-01",
            "2026-05-25", "2026-05-27", "2026-05-29", "2026-06-16", "2026-06-26"]
RAMADAN = ("2026-02-18", "2026-03-19")
# Wikipedia "2026 in Lebanon" / "Timeline of the 2026 Lebanon war" (retrieved 2026-10-17).
# Retrospectively selected; exposure of each branch is unverified.
WAR = ("2026-03-02", "2026-04-16")
BEIRUT_STRIKES = ["2026-03-03", "2026-03-08", "2026-03-12", "2026-03-18", "2026-03-31", "2026-04-01",
                  "2026-04-05", "2026-04-08", "2026-04-09", "2026-05-06", "2026-06-07"]
WEATHER_FILES = {"Beirut": "beirut_weather_2026H1.json", "Batroun": "batroun_weather_2026H1.json",
                 "Broummana": "broummana_weather_2026H1.json"}

COLD = ["Cold Coffee", "Cold Drinks"]
HOT = ["Hot Coffee"]


# %% Data: branch x day targets (all raw lines retained; orders = distinct Transaction IDs)
raw = pd.read_csv(DATA)
raw["Date"] = pd.to_datetime(raw["Date"])
DAYS = pd.date_range(raw["Date"].min(), raw["Date"].max())
BRANCHES = sorted(raw["Branch"].unique())
idx = pd.MultiIndex.from_product([BRANCHES, DAYS], names=["branch", "date"])


def daily(frame, how):
    g = frame.groupby(["Branch", "Date"])
    s = g["Transaction ID"].nunique() if how == "orders" else g["Quantity"].sum()
    return s.rename_axis(["branch", "date"]).reindex(idx, fill_value=0).astype(float)


TARGETS = {
    "orders": daily(raw, "orders"),
    "cold_drink_units": daily(raw[raw["Category"].isin(COLD)], "units"),
    "hot_coffee_units": daily(raw[raw["Category"].isin(HOT)], "units"),
    "takeaway_orders": daily(raw[raw["Order Type"] == "Takeaway"], "orders"),
}


# %% Factors: one daily series per factor (same value for all branches on a date)
def zscore(s):
    return (s - s.mean()) / s.std()


factors = {}
for loc, fname in WEATHER_FILES.items():
    w = pd.DataFrame(json.loads((EXT / fname).read_text())["daily"])
    w = w.set_index(pd.to_datetime(w.pop("time"))).reindex(DAYS)
    factors[f"{loc}: max temperature"] = zscore(w["temperature_2m_max"])
    factors[f"{loc}: rain (log mm)"] = zscore(np.log1p(w["precipitation_sum"]))
    factors[f"{loc}: rainy day (>1 mm)"] = (w["precipitation_sum"] > 1).astype(float)
    factors[f"{loc}: max wind"] = zscore(w["wind_speed_10m_max"])
factors["Public holiday"] = pd.Series(DAYS.isin(pd.to_datetime(HOLIDAYS)).astype(float), DAYS)
factors["Ramadan"] = pd.Series(((DAYS >= RAMADAN[0]) & (DAYS <= RAMADAN[1])).astype(float), DAYS)
factors["War period (2 Mar-16 Apr)"] = pd.Series(((DAYS >= WAR[0]) & (DAYS <= WAR[1])).astype(float), DAYS)
factors["Beirut strike day"] = pd.Series(DAYS.isin(pd.to_datetime(BEIRUT_STRIKES)).astype(float), DAYS)

# Which targets each factor is tested on (a mechanism, not a fishing trip).
PLAN = {}
for name in factors:
    if "temperature" in name:
        PLAN[name] = ["orders", "cold_drink_units", "hot_coffee_units"]
    elif "rain" in name or "wind" in name:
        PLAN[name] = ["orders", "takeaway_orders"]
    elif name in ("War period (2 Mar-16 Apr)", "Beirut strike day"):
        PLAN[name] = ["orders", "takeaway_orders"]
    else:
        PLAN[name] = ["orders"]


# %% Model: Poisson GLM by IRLS (branch, branch x weekday, branch x trend [+ factor])
def design(extra=None):
    b = idx.get_level_values("branch")
    d = idx.get_level_values("date")
    t = ((d - DAYS[0]).days / len(DAYS)).to_numpy()
    cols = []
    for br in BRANCHES:
        m = (b == br).astype(float)
        cols.append(m)
        cols.append(m * t)
        for k in range(1, 7):
            cols.append(m * (d.dayofweek == k))
    X = np.column_stack(cols).astype(float)
    if extra is not None:
        X = np.column_stack([X, extra])
    return X


def fit_poisson(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    beta[:] = 0
    eta = np.log(np.maximum(y, 0.5))
    pen = RIDGE * np.eye(X.shape[1])
    for _ in range(iters):
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X + pen, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
        eta = np.clip(X @ beta, -20, 20)
    return beta


def deviance(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


DATES = idx.get_level_values("date")
HALF = {"Jan-Mar -> Apr-Jun": (DATES < SPLIT, DATES >= SPLIT),
        "Apr-Jun -> Jan-Mar": (DATES >= SPLIT, DATES < SPLIT)}
X0 = design()


def factor_column(series):
    return series.reindex(DATES).to_numpy()


def holdout(y, xcol, train, test):
    """Held-out deviance of baseline and baseline + factor, plus the factor coefficient."""
    b0 = fit_poisson(X0[train], y[train])
    X1 = np.column_stack([X0, xcol])
    b1 = fit_poisson(X1[train], y[train])
    d0 = deviance(y[test], np.exp(X0[test] @ b0))
    d1 = deviance(y[test], np.exp(X1[test] @ b1))
    return d0, d1, b1[-1]


# Baseline deviances per target/direction are reused by the placebo loop
_base = {}


def gain(y_name, y, xcol, direction):
    train, test = HALF[direction]
    key = (y_name, direction)
    if key not in _base:
        b0 = fit_poisson(X0[train], y[train])
        _base[key] = deviance(y[test], np.exp(X0[test] @ b0))
    X1 = np.column_stack([X0, xcol])
    b1 = fit_poisson(X1[train], y[train])
    d1 = deviance(y[test], np.exp(X1[test] @ b1))
    return (_base[key] - d1) / _base[key], b1[-1]


def shifted(series, k):
    return pd.Series(np.roll(series.to_numpy(), k), series.index)


# %% Run: held-out gain + placebo for every factor x target
rows = []
null_cache = {}
for fname, series in factors.items():
    for tname in PLAN[fname]:
        y = TARGETS[tname].to_numpy()
        rec = {"factor": fname, "target": tname}
        for direction in HALF:
            g, coef = gain(tname, y, factor_column(series), direction)
            null = np.array([gain(tname, y, factor_column(shifted(series, k)), direction)[0] for k in SHIFTS])
            null_cache[(fname, tname, direction)] = null
            tag = "fwd" if direction.startswith("Jan") else "rev"
            rec[f"gain_{tag}"] = g
            rec[f"effect_{tag}"] = float(np.exp(coef) - 1)  # per 1 SD (continuous) or on flagged days
            rec[f"p_{tag}"] = (1 + np.sum(null >= g)) / (1 + len(null))
            rec[f"null95_{tag}"] = float(np.quantile(null, 0.95))
        rows.append(rec)
        print(f"{fname:<34} {tname:<17} gain fwd {rec['gain_fwd']:+.2%} (p={rec['p_fwd']:.3f}) "
              f"rev {rec['gain_rev']:+.2%} (p={rec['p_rev']:.3f})  effect {rec['effect_fwd']:+.1%}")

res = pd.DataFrame(rows)
# Holm correction on the larger of the two p-values (a factor must pass both directions)
res["p_both"] = res[["p_fwd", "p_rev"]].max(axis=1)
order = res["p_both"].sort_values().index
m = len(res)
adj = np.maximum.accumulate([(m - i) * res.loc[j, "p_both"] for i, j in enumerate(order)])
res.loc[order, "p_holm"] = np.minimum(adj, 1)
res["passes"] = (res["gain_fwd"] > 0) & (res["gain_rev"] > 0) & (res["p_holm"] < 0.05)
res["effect_sign_stable"] = np.sign(res["effect_fwd"]) == np.sign(res["effect_rev"])


# %% Positive control: plant a known effect and see whether the test finds it
def plant(y, flag, effect):
    """Remove `effect` share of units on flagged days by binomial thinning."""
    keep = np.where(flag > 0, 1 - effect, 1.0)
    return RNG.binomial(y.astype(int), keep).astype(float)


power = []
y_orders = TARGETS["orders"].to_numpy()
for fname in ["Beirut: rainy day (>1 mm)", "Public holiday", "Beirut strike day"]:
    flag = factor_column(factors[fname])
    nf = factors[fname]
    for effect in [0.05, 0.10, 0.20]:
        hits = 0
        for _ in range(N_SIM):
            yp = plant(y_orders, flag, effect)
            ok = True
            for direction in HALF:
                train, test = HALF[direction]
                d0, d1, _ = holdout(yp, flag, train, test)
                g = (d0 - d1) / d0
                # Placebo threshold re-estimated on the planted data with a thinned set of offsets
                nulls = []
                for k in SHIFTS[:: max(1, len(SHIFTS) // 20)]:
                    s0, s1, _ = holdout(yp, factor_column(shifted(nf, k)), train, test)
                    nulls.append((s0 - s1) / s0)
                ok &= g > 0 and g > np.quantile(nulls, 0.95)
            hits += ok
        power.append({"factor": fname, "planted_drop": effect, "detection_rate": hits / N_SIM,
                      "flagged_days": int(nf.sum())})
        print(f"positive control {fname:<28} planted -{effect:.0%}: detected {hits / N_SIM:.0%}")
power = pd.DataFrame(power)


# %% Save outputs
OUT.mkdir(exist_ok=True)
res.to_csv(OUT / "external_factor_test.csv", index=False, float_format="%.6g")
meta = {
    "question": "Does each external factor improve out-of-sample prediction of sales beyond a transactions-only baseline?",
    "protocol": {
        "unit": "branch x day",
        "baseline": "Poisson GLM: branch + branch x weekday + branch x linear trend (ridge 1e-4)",
        "directions": list(HALF),
        "split_date": str(SPLIT.date()),
        "metric": "relative reduction in held-out mean Poisson deviance",
        "placebo": f"circular shifts of the factor by {SHIFTS[0]}-{SHIFTS[-1]} days ({len(SHIFTS)} offsets)",
        "pass_rule": "gain > 0 in both directions and Holm-adjusted max(p_fwd, p_rev) < 0.05",
        "positive_control": f"binomial thinning on flagged days, {N_SIM} simulations per cell",
        "raw_lines": "all retained; orders counted as distinct Transaction IDs",
    },
    "limits": [
        "Weather is realised weather (perfect-knowledge retrospective test), not a forecast-time input.",
        "Conflict and strike dates were selected after the fact; branch exposure is unverified.",
        "Branch locations are unverified; weather is tested at three candidate locations.",
        "Passing shows predictive association, not causation; failing does not prove zero effect "
        "(see positive_control for detectable sizes).",
        "These factors are not inputs to the strict model (src/strict_model.py).",
    ],
    "results": res.to_dict(orient="records"),
    "positive_control": power.to_dict(orient="records"),
    "provenance": {
        "run_date": str(date.today()),
        "transactions_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "weather_source": "Open-Meteo historical archive API (daily), files in data/external/",
        "event_sources": ["https://en.wikipedia.org/wiki/2026_in_Lebanon",
                          "https://en.wikipedia.org/wiki/Timeline_of_the_2026_Lebanon_war"],
        "holiday_source": "notebooks/03_holidays.ipynb HOLIDAYS list",
        "quick_mode": QUICK,
        "numpy": np.__version__, "pandas": pd.__version__,
    },
}
(OUT / "external_factor_test.json").write_text(json.dumps(meta, indent=2, default=float))
(ROOT / "app").mkdir(exist_ok=True)
(ROOT / "app" / "evidence_data.js").write_text(
    "// Generated by src/external_factor_test.py. Do not edit by hand; rerun the script.\n"
    f"window.EVIDENCE_DATA = {json.dumps(meta, default=float)};\n")
print(f"\nsaved reports/external_factor_test.csv and .json  ({int(res['passes'].sum())} of {len(res)} tests pass)")


# %% Chart: held-out gain vs placebo band, and positive-control power
INK2, GRID, SURFACE, ACCENT, WARN, MUTED = "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#d64545", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
show = res[res["target"].isin(["orders", "cold_drink_units"])].copy()
show["label"] = show["factor"] + "  ->  " + show["target"].str.replace("_", " ")
show = show.sort_values("gain_fwd")
fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(show) + 2), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
yy = np.arange(len(show))
XMIN = -0.05  # clip large losses so the small gains stay readable
for tag, off, color, lab in [("fwd", -0.18, ACCENT, "Train Jan-Mar, test Apr-Jun"),
                             ("rev", 0.18, "#9ec5f4", "Train Apr-Jun, test Jan-Mar")]:
    vals = show[f"gain_{tag}"].to_numpy()
    ax.barh(yy + off, np.maximum(vals, XMIN), height=0.34, color=color, label=lab)
    for i, v in enumerate(vals):
        if v < XMIN:
            ax.text(XMIN, i + off, f"{v:+.0%} ", va="center", ha="right", fontsize=7.5, color=INK2)
    ax.scatter(show[f"null95_{tag}"], yy + off, marker="|", s=80, color=INK2, zorder=3,
               label="95th pct of placebo (wrong dates)" if tag == "fwd" else None)
right = max(show[["gain_fwd", "gain_rev", "null95_fwd", "null95_rev"]].max().max(), 0.01) + 0.012
for i, (ok, p) in enumerate(zip(show["passes"], show["p_holm"])):
    ax.text(right, i, "PASS" if ok else f"Holm p={p:.2f}", va="center", fontsize=7.5,
            color=WARN if ok else INK2, fontweight="bold" if ok else "normal")
ax.set_xlim(XMIN - 0.012, right + 0.022)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, show["label"], fontsize=8.5)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1%}"))
ax.set_xlabel("Reduction in held-out prediction error vs transactions-only baseline")
ax.set_title(f"Do outside factors predict unseen days? {int(res['passes'].sum())} of {len(res)} tests pass")
ax.legend(loc="upper left", fontsize=8)
ax = axes[1]
for fname, g in power.groupby("factor"):
    ax.plot(g["planted_drop"], g["detection_rate"], marker="o", label=f"{fname} ({g['flagged_days'].iloc[0]} days)")
ax.axhline(0.8, color=MUTED, ls="--", lw=1)
ax.set_ylim(0, 1.05)
ax.set_xticks([0.05, 0.10, 0.20])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"-{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Planted drop in orders on flagged days")
ax.set_ylabel("Share of simulations detected")
ax.set_title("Positive control: could the test see a real effect?")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "external_factor_holdout.png", dpi=150, bbox_inches="tight")
print("saved reports/figures/external_factor_holdout.png")
