"""
Why do Stories' numbers move? Driver decompositions, each checked in both halves of the year.

1. Why is gross profit growing?
   Exact log decomposition of monthly gross profit:
       GP = orders x items per order x revenue per item x margin %
       orders = identified-customer orders (active customers x visits each) + walk-in orders
   Jan->Jun, and separately Jan->Mar and Apr->Jun (does the same driver dominate both halves?).
2. Why does Garden earn a lower margin than Harbour?
   Margin gap = price effect (Garden's lower prices) + cost effect (Garden's higher unit costs)
              + mix effect (Garden selling a different product mix), at Garden's own quantities.
3. Why do daily orders go up and down?
   Nested Poisson models fitted on one half and scored on the other (both directions):
   branch -> + growth trend -> + weekday. Held-out R^2 of each step, compared with the ceiling set
   by Poisson counting noise (the randomness of individual customers arriving even when the
   expected rate is known). Whatever is left above that noise is the only room for anything else
   (weather, events, ...) to explain.
4. What is different about Tuesday?
   Tuesday multiplier by branch, half, daypart, category, order type and customer type. A uniform
   multiplier everywhere points to a chain-wide cause (calendar, offer, operations, recording), not
   to a type of customer or product. The data cannot name the cause; the company can.

These are accounting identities and held-out fits, not causal effects. All raw lines are retained.

Run from the repo root:
    python src/why_drivers.py
Outputs:
    reports/why_drivers.json
    app/why_data.js                    (same numbers, loaded by app/planner.html "Why" tab)
    reports/figures/why_drivers.png

Requires: pandas, numpy, matplotlib
"""

# %% Setup
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
R, GP = "Line Revenue (USD)", "Line Gross Profit (USD)"
SPLIT = pd.Timestamp("2026-04-01")
HALVES = {"Jan-Mar": lambda d: d < SPLIT, "Apr-Jun": lambda d: d >= SPLIT}

df = pd.read_csv(ROOT / "data" / "stories_transactions.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.month
df["hour"] = df["Time"].str[:2].astype(int)
df["daypart"] = pd.cut(df["hour"], [0, 10, 17, 24], include_lowest=True,
                       labels=["morning", "afternoon", "evening"]).astype(str)
o = df.groupby("Transaction ID").agg(
    date=("Date", "first"), month=("month", "first"), branch=("Branch", "first"), daypart=("daypart", "first"),
    order_type=("Order Type", "first"), customer=("Customer ID", "first"),
    gp=(GP, "sum"), rev=(R, "sum"), units=("Quantity", "sum"))
out = {"run_date": str(date.today()), "split": str(SPLIT.date()), "raw_lines_retained": True}


# %% 1. Why is gross profit growing?
known = o[o["customer"].notna()]
M = pd.DataFrame({
    "gp": o.groupby("month")["gp"].sum(), "orders": o.groupby("month").size(),
    "known": known.groupby("month").size(), "walkin": o[o["customer"].isna()].groupby("month").size(),
    "active": known.groupby("month")["customer"].nunique(),
    "units": o.groupby("month")["units"].sum(), "rev": o.groupby("month")["rev"].sum()})
M["visits"] = M["known"] / M["active"]
M["items_per_order"] = M["units"] / M["orders"]
M["rev_per_item"] = M["rev"] / M["units"]
M["margin"] = M["gp"] / M["rev"]


def logmean(x, y):
    return x if x == y else (y - x) / np.log(y / x)


def growth_decomp(a, b):
    A, B = M.loc[a], M.loc[b]
    total = np.log(B["gp"] / A["gp"])
    wk = logmean(A["known"], B["known"]) / logmean(A["orders"], B["orders"])
    ww = logmean(A["walkin"], B["walkin"]) / logmean(A["orders"], B["orders"])
    parts = {
        "More identified customers active": wk * np.log(B["active"] / A["active"]),
        "More visits per identified customer": wk * np.log(B["visits"] / A["visits"]),
        "More walk-in orders": ww * np.log(B["walkin"] / A["walkin"]),
        "More items per order": np.log(B["items_per_order"] / A["items_per_order"]),
        "Higher price / richer mix per item": np.log(B["rev_per_item"] / A["rev_per_item"]),
        "Higher margin %": np.log(B["margin"] / A["margin"]),
    }
    return {"from_month": int(a), "to_month": int(b), "gp_from": round(float(A["gp"]), 2),
            "gp_to": round(float(B["gp"]), 2), "growth": float(np.exp(total) - 1),
            "share_of_growth": {k: float(v / total) for k, v in parts.items()}}


out["growth"] = {"Jan->Jun": growth_decomp(1, 6), "Jan->Mar": growth_decomp(1, 3), "Apr->Jun": growth_decomp(4, 6),
                 "monthly": M[["gp", "orders", "active", "visits", "walkin", "items_per_order", "margin"]]
                 .round(4).reset_index().to_dict(orient="records")}
print("=== 1. Profit growth drivers (share of log growth) ===")
for k in ["Jan->Jun", "Jan->Mar", "Apr->Jun"]:
    g = out["growth"][k]
    print(f"{k}: GP {g['growth']:+.1%} | " + ", ".join(f"{n} {s:+.0%}" for n, s in g["share_of_growth"].items()))


# %% 2. Why does Garden earn less than Harbour?
def margin_gap(d, branch="Garden", ref="Harbour"):
    u = d.groupby(["Branch", "Product"]).agg(p=("Unit Price (USD)", "first"), c=("Unit Cost (USD)", "first"),
                                             q=("Quantity", "sum"))
    G, H = u.loc[branch], u.loc[ref]
    q = G["q"]
    m_branch = (q * (G["p"] - G["c"])).sum() / (q * G["p"]).sum()
    m_ref = (H["q"] * (H["p"] - H["c"])).sum() / (H["q"] * H["p"]).sum()
    m_ref_price = (q * (H["p"] - G["c"])).sum() / (q * H["p"]).sum()
    m_ref_both = (q * (H["p"] - H["c"])).sum() / (q * H["p"]).sum()
    return {"branch_margin": float(m_branch), "reference_margin": float(m_ref), "gap_pts": float(100 * (m_branch - m_ref)),
            "price_effect_pts": float(100 * (m_branch - m_ref_price)),
            "cost_effect_pts": float(100 * (m_ref_price - m_ref_both)),
            "mix_effect_pts": float(100 * (m_ref_both - m_ref))}


out["branch_margin_gap"] = {}
for br in ["Garden", "Uptown"]:
    out["branch_margin_gap"][br] = {"all": margin_gap(df, br)} | {
        h: margin_gap(df[f(df["Date"])], br) for h, f in HALVES.items()}
print("\n=== 2. Margin gap vs Harbour (percentage points) ===")
for br, v in out["branch_margin_gap"].items():
    for h, g in v.items():
        print(f"{br:7s} {h:8s} gap {g['gap_pts']:+.2f} = price {g['price_effect_pts']:+.2f} "
              f"+ cost {g['cost_effect_pts']:+.2f} + mix {g['mix_effect_pts']:+.2f}")


# %% 3. Why do daily orders go up and down? Held-out explained variance vs noise ceiling
DAYS = pd.date_range(df["Date"].min(), df["Date"].max())
BR = sorted(df["Branch"].unique())
idx = pd.MultiIndex.from_product([BR, DAYS], names=["branch", "date"])
y = o.groupby(["branch", "date"]).size().reindex(idx, fill_value=0).to_numpy(float)
bb, dd = idx.get_level_values("branch"), idx.get_level_values("date")
tt = ((dd - DAYS[0]).days / len(DAYS)).to_numpy()


def design(level):
    cols = []
    for br in BR:
        m = (bb == br).astype(float)
        cols.append(m)
        if level >= 2:
            cols.append(m * tt)
        if level >= 3:
            cols += [m * (dd.dayofweek == k) for k in range(1, 7)]
    return np.column_stack(cols)


def fit_poisson(X, yv):
    beta, eta = np.zeros(X.shape[1]), np.log(np.maximum(yv, 0.5))
    for _ in range(60):
        mu = np.exp(eta)
        new = np.linalg.solve((X.T * mu) @ X + 1e-6 * np.eye(X.shape[1]), (X.T * mu) @ (eta + (yv - mu) / mu))
        done = np.abs(new - beta).max() < 1e-9
        beta, eta = new, X @ new
        if done:
            break
    return beta


STEPS = [(1, "Branch"), (2, "+ growth trend"), (3, "+ weekday")]
out["daily_variation"] = {}
for name, tr, te in [("Jan-Mar -> Apr-Jun", dd < SPLIT, dd >= SPLIT), ("Apr-Jun -> Jan-Mar", dd >= SPLIT, dd < SPLIT)]:
    yt = y[te]
    sst = ((yt - yt.mean()) ** 2).sum()
    steps = []
    for lev, lab in STEPS:
        X = design(lev)
        mu = np.exp(X[te] @ fit_poisson(X[tr], y[tr]))
        steps.append({"model": lab, "heldout_r2": float(1 - ((yt - mu) ** 2).sum() / sst)})
    resid_var = float(((yt - mu) ** 2).mean())
    noise_var = float(mu.mean())  # Poisson variance = mean
    out["daily_variation"][name] = {
        "steps": steps, "residual_variance": resid_var, "poisson_noise_variance": noise_var,
        "residual_over_noise": resid_var / noise_var,
        "ceiling_r2_if_rest_is_noise": float(1 - noise_var * len(yt) / sst),
        "room_left_for_other_factors_r2": float(max(0.0, (resid_var - noise_var) * len(yt) / sst)),
    }
print("\n=== 3. Daily orders: held-out R^2 by step, vs counting-noise ceiling ===")
for name, v in out["daily_variation"].items():
    print(name, " -> ".join(f"{s['model']} {s['heldout_r2']:.3f}" for s in v["steps"]),
          f"| ceiling {v['ceiling_r2_if_rest_is_noise']:.3f}, residual/noise {v['residual_over_noise']:.2f}, "
          f"room left {v['room_left_for_other_factors_r2']:.3f}")


# %% 4. What is different about Tuesday?
o["dow"] = o["date"].dt.dayofweek
o["half"] = np.where(o["date"] < SPLIT, "Jan-Mar", "Apr-Jun")
o["customer_type"] = np.where(o["customer"].notna(), "identified", "walk-in")
n_days = o.groupby("dow")["date"].nunique()


def tuesday_multiplier(group_col):
    per_day = o.groupby([group_col, "dow"]).size().unstack() / n_days
    return (per_day[1] / per_day[[0, 2, 3, 4]].mean(axis=1)).round(3).to_dict()


line_dow = df["Date"].dt.dayofweek
cat_day = df.groupby([df["Category"], line_dow])["Quantity"].sum().unstack() / n_days
tue = {
    "by_branch": tuesday_multiplier("branch"), "by_half": tuesday_multiplier("half"),
    "by_daypart": tuesday_multiplier("daypart"), "by_order_type": tuesday_multiplier("order_type"),
    "by_customer_type": tuesday_multiplier("customer_type"),
    "by_category": (cat_day[1] / cat_day[[0, 2, 3, 4]].mean(axis=1)).round(3).to_dict(),
    "avg_order_value": {"Tuesday": round(float(o.loc[o["dow"] == 1, "rev"].mean()), 2),
                        "other weekdays": round(float(o.loc[o["dow"].isin([0, 2, 3, 4]), "rev"].mean()), 2)},
    "tuesdays_above_every_other_day_median": None,
}
daily_total = o.groupby("date").size()
other_median = daily_total[daily_total.index.dayofweek != 1].median()
tue["tuesdays_above_every_other_day_median"] = f"{int((daily_total[daily_total.index.dayofweek == 1] > other_median).sum())} of {int((daily_total.index.dayofweek == 1).sum())}"
allv = [v for k in ["by_branch", "by_half", "by_daypart", "by_order_type", "by_customer_type", "by_category"]
        for v in tue[k].values()]
tue["range_of_multipliers"] = [float(min(allv)), float(max(allv))]
out["tuesday"] = tue
print(f"\n=== 4. Tuesday multiplier: {min(allv):.2f}-{max(allv):.2f} across every cut ===")
for k in ["by_branch", "by_half", "by_daypart", "by_order_type", "by_customer_type"]:
    print(f"  {k}: {tue[k]}")
print(f"  AOV {tue['avg_order_value']} | Tuesdays above other-day median: {tue['tuesdays_above_every_other_day_median']}")


# %% Save
(ROOT / "reports" / "why_drivers.json").write_text(json.dumps(out, indent=2))
(ROOT / "app").mkdir(exist_ok=True)
(ROOT / "app" / "why_data.js").write_text(
    "// Generated by src/why_drivers.py. Do not edit by hand; rerun the script.\n"
    f"window.WHY_DATA = {json.dumps(out)};\n")
print("\nsaved reports/why_drivers.json and app/why_data.js")


# %% Chart
INK2, GRID, SURFACE, ACCENT, LIGHT, MUTED, WARN = "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#9ec5f4", "#b9b8b3", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
labels = list(out["growth"]["Jan->Jun"]["share_of_growth"])
yy = np.arange(len(labels))
for i, (k, c) in enumerate([("Jan->Mar", ACCENT), ("Apr->Jun", LIGHT)]):
    vals = [out["growth"][k]["share_of_growth"][l] for l in labels]
    ax.barh(yy + (i - 0.5) * 0.36, vals, height=0.36, color=c, label=f"{k} (GP {out['growth'][k]['growth']:+.0%})")
ax.set_yticks(yy, labels)
ax.invert_yaxis()
ax.axvline(0, color=INK2, lw=0.8)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Share of gross-profit growth")
ax.set_title("Why profit grows: more people, more visits")
ax.legend(loc="lower right")

ax = axes[0, 1]
g = out["branch_margin_gap"]["Garden"]["all"]
parts = [("Lower prices", g["price_effect_pts"]), ("Higher unit costs", g["cost_effect_pts"]),
         ("Different product mix", g["mix_effect_pts"])]
rows_g = parts + [("Total gap", g["gap_pts"])]
for i, (lab, v) in enumerate(rows_g):
    color = "#0b0b0b" if lab == "Total gap" else (WARN if lab == "Higher unit costs" else MUTED)
    ax.barh(i, -v, color=color, height=0.6)
    share = "" if lab == "Total gap" else f"  ({v / g['gap_pts']:.0%})"
    ax.text(-v + 0.3, i, f"{-v:.1f} pts{share}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(rows_g)), [r[0] for r in rows_g])
ax.invert_yaxis()
ax.set_xlim(0, -g["gap_pts"] * 1.3)
ax.set_xlabel(f"Margin points below Harbour (Garden {g['branch_margin']:.0%} vs Harbour {g['reference_margin']:.0%})")
ax.set_title(f"Why Garden earns less: {g['cost_effect_pts'] / g['gap_pts']:.0%} is unit cost")
ax.grid(axis="y", visible=False)

ax = axes[1, 0]
for i, (name, c) in enumerate([("Jan-Mar -> Apr-Jun", ACCENT), ("Apr-Jun -> Jan-Mar", LIGHT)]):
    v = out["daily_variation"][name]
    r2 = [s["heldout_r2"] for s in v["steps"]]
    ax.plot([s["model"] for s in v["steps"]], r2, marker="o", color=c, label=f"train {name.split(' -> ')[0]}, test {name.split(' -> ')[1]}")
    ax.axhline(v["ceiling_r2_if_rest_is_noise"], color=c, ls="--", lw=1)
ax.text(0.02, out["daily_variation"]["Jan-Mar -> Apr-Jun"]["ceiling_r2_if_rest_is_noise"] + 0.015,
        "dashed = most any model could explain (rest is random arrivals)", fontsize=8, color=INK2)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Daily orders explained on unseen months")
ax.set_title("Why daily sales move: weekday + growth, then chance")
ax.legend(loc="lower right")

ax = axes[1, 1]
cuts = [("branch", "by_branch"), ("half", "by_half"), ("daypart", "by_daypart"), ("order type", "by_order_type"),
        ("customer", "by_customer_type"), ("category", "by_category")]
xs, ys, cs = [], [], []
for j, (lab, key) in enumerate(cuts):
    for v in tue[key].values():
        xs.append(j)
        ys.append(v)
ax.scatter(np.array(xs) + np.random.default_rng(0).uniform(-0.12, 0.12, len(xs)), ys, color=ACCENT, s=28, alpha=0.8)
ax.axhline(1, color=MUTED, lw=1)
ax.set_xticks(range(len(cuts)), [c[0] for c in cuts])
ax.set_ylim(0.8, 2.0)
ax.set_ylabel("Tuesday / average Mon, Wed-Fri")
ax.set_title(f"Tuesday is ~{np.median(allv):.2f}x in every slice")
fig.tight_layout()
(ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "reports" / "figures" / "why_drivers.png", dpi=150, bbox_inches="tight")
print("saved reports/figures/why_drivers.png")
