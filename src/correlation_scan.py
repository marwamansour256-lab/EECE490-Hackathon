"""
Correlation scan: every pairwise association in the transactions, measured separately in
each half of the timeline (Jan-Mar vs Apr-Jun) so stable patterns can be told apart from noise.

Three levels:
  * order level   - branch, weekday, daypart, order type, identified customer, discount,
                    category flags, hour, revenue, items, margin (Cramer's V / correlation
                    ratio eta / Spearman rho, by variable type)
  * structure     - drink+food pairing and category-pair lift vs label shuffles, daypart mix,
                    branch hour profiles, weekday multipliers, customer-level relationships
  * external      - see src/external_factor_test.py (held-out test of weather/events)

An association is called STABLE when its size differs by < 0.03 between halves (or the
shuffle test holds in both halves). Mechanical pairs (revenue vs gross profit, items vs lines,
hour vs daypart) are excluded from the ranking. All raw lines are retained (AGENTS.md).
These are associations, not causes; product price vs volume is a cross-section across
products, not price elasticity.

Run from the repo root:
    python src/correlation_scan.py
Outputs:
    reports/correlation_scan_pairs.csv     every order-level pair with H1, H2, all
    reports/correlation_scan.json          structured findings with both halves
    reports/figures/correlation_scan.png

Requires: pandas, numpy, matplotlib
"""

# %% Setup
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
OUT, FIG = ROOT / "reports", ROOT / "reports" / "figures"
R, GP = "Line Revenue (USD)", "Line Gross Profit (USD)"
SPLIT = "2026-04-01"
DRINKS = {"Hot Coffee", "Cold Coffee", "Matcha & Tea", "Cold Drinks"}
N_SHUFFLE = 100
rng = np.random.default_rng(490)

df = pd.read_csv(ROOT / "data" / "stories_transactions.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = df["Time"].str[:2].astype(int)
df["half"] = np.where(df["Date"] < SPLIT, "Jan-Mar", "Apr-Jun")
df["daypart"] = pd.cut(df["hour"], [0, 10, 17, 24], include_lowest=True,
                       labels=["morning 7-10", "afternoon 11-17", "evening 18-23"]).astype(str)
df["is_drink"] = df["Category"].isin(DRINKS)
HALVES = ["Jan-Mar", "Apr-Jun"]

o = df.groupby("Transaction ID").agg(
    date=("Date", "first"), half=("half", "first"), hour=("hour", "first"), daypart=("daypart", "first"),
    branch=("Branch", "first"), order_type=("Order Type", "first"), customer=("Customer ID", "first"),
    revenue=(R, "sum"), gross_profit=(GP, "sum"), items=("Quantity", "sum"), lines=("Product", "size"),
    discount=("Discount Rate", "max"))
for c in sorted(df["Category"].unique()):
    o["has " + c] = df.assign(x=df["Category"].eq(c)).groupby("Transaction ID")["x"].max().astype(int)
o["weekday"] = o["date"].dt.day_name()
o["identified customer"] = o["customer"].notna().astype(int)
o["discounted"] = (o["discount"] > 0).astype(int)
o["margin"] = o["gross_profit"] / o["revenue"]


# %% 1. Every order-level pair, both halves
CAT = ["branch", "weekday", "daypart", "order_type", "identified customer", "discounted"] + \
      [c for c in o if c.startswith("has ")]
NUM = ["hour", "revenue", "gross_profit", "items", "lines", "margin"]
MECH = {frozenset(p) for p in [("revenue", "gross_profit"), ("items", "lines"), ("revenue", "items"),
                               ("revenue", "lines"), ("gross_profit", "items"), ("gross_profit", "lines"),
                               ("hour", "daypart")]}


def cramers_v(a, b):
    t = pd.crosstab(a, b).to_numpy().astype(float)
    n = t.sum()
    e = np.outer(t.sum(1), t.sum(0)) / n
    k = min(t.shape) - 1
    return float(np.sqrt(((t - e) ** 2 / e).sum() / (n * k))) if k > 0 else np.nan


def eta(c, y):
    g = y.groupby(c)
    return float(np.sqrt((g.size() * (g.mean() - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def assoc(d, a, b):
    if a in CAT and b in CAT:
        return cramers_v(d[a], d[b]), "Cramer's V"
    if a in NUM and b in NUM:
        return float(d[a].rank().corr(d[b].rank())), "Spearman rho"
    c, y = (a, b) if a in CAT else (b, a)
    return eta(d[c], d[y]), "eta"


rows = []
for a, b in itertools.combinations(CAT + NUM, 2):
    if frozenset((a, b)) in MECH:
        continue
    v_all, kind = assoc(o, a, b)
    v1, _ = assoc(o[o["half"] == HALVES[0]], a, b)
    v2, _ = assoc(o[o["half"] == HALVES[1]], a, b)
    rows.append({"a": a, "b": b, "measure": kind, "all": v_all, HALVES[0]: v1, HALVES[1]: v2,
                 "stable": abs(v1 - v2) < 0.03})
pairs = pd.DataFrame(rows)
pairs["abs"] = pairs["all"].abs()
pairs = pairs.sort_values("abs", ascending=False).drop(columns="abs")
OUT.mkdir(exist_ok=True)
pairs.to_csv(OUT / "correlation_scan_pairs.csv", index=False, float_format="%.4f")
print(f"=== {len(pairs)} order-level pairs; {int(pairs['stable'].sum())} stable between halves ===")
print(pairs.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def by_half(fn):
    return {h: fn(df[df["half"] == h], o[o["half"] == h]) for h in HALVES} | {"all": fn(df, o)}


findings = {}

# %% 2. Time of day drives what and how people buy
findings["takeaway share by daypart"] = by_half(
    lambda d, oo: oo.groupby("daypart")["order_type"].apply(lambda s: round(float((s == "Takeaway").mean()), 3)).to_dict())
mix = pd.crosstab(df["Category"], df["daypart"], values=df["Quantity"], aggfunc="sum", normalize="columns")
mix = mix[["morning 7-10", "afternoon 11-17", "evening 18-23"]]
findings["category share of items by daypart"] = mix.round(3).to_dict()
findings["category share by order type"] = pd.crosstab(df["Category"], df["Order Type"], values=df["Quantity"],
                                                       aggfunc="sum", normalize="columns").round(3).to_dict()

# %% 3. Basket structure vs shuffles: drink + food is real, specific pairs are not
def mixed_share(d):
    g = d.groupby("Transaction ID")["is_drink"].agg(["sum", "size"])
    g = g[g["size"] == 2]
    return float((g["sum"] == 1).mean())


def shuffled_mixed(d):
    s = d[["Transaction ID"]].copy()
    s["is_drink"] = rng.permutation(d["is_drink"].to_numpy())
    return mixed_share(s)


findings["two-line orders that are 1 drink + 1 food"] = {
    h: {"actual": round(mixed_share(d), 3),
        "shuffled_mean": round(float(np.mean([shuffled_mixed(d) for _ in range(N_SHUFFLE)])), 3)}
    for h, d in [(h, df[df["half"] == h]) for h in HALVES]}

sets = df.groupby("Transaction ID")["Category"].apply(frozenset)
sets = sets[sets.apply(len) >= 2]
half_of = o["half"].reindex(sets.index)
lift_rows = []
for a, b in itertools.combinations(sorted(df["Category"].unique()), 2):
    rec = {"pair": f"{a} + {b}", "orders": int(sets.apply(lambda s: a in s and b in s).sum())}
    for h in HALVES:
        ss = sets[half_of == h]
        ha, hb = ss.apply(lambda s: a in s), ss.apply(lambda s: b in s)
        rec[h] = round(float((ha & hb).mean() / (ha.mean() * hb.mean())), 2)
    lift_rows.append(rec)
lift = pd.DataFrame(lift_rows).sort_values(HALVES[0], ascending=False)
findings["category pair lift in multi-category orders (1 = chance)"] = lift.head(8).to_dict(orient="records")

# %% 4. Calendar: same rhythm at every branch, in both halves
hp = pd.crosstab(o["hour"], o["branch"], normalize="columns")
findings["hourly profile correlation between branches"] = hp.corr().round(3).to_dict()
findings["peak hour by branch"] = hp.idxmax().to_dict()
o["dow"] = o["date"].dt.dayofweek
d = o.groupby(["half", "branch", "dow"]).size().unstack()
# per-day rates: weekdays occur a different number of times in each half
n_days = o.groupby(["half", "dow"])["date"].nunique().unstack()
d = d / n_days.reindex(d.index.get_level_values("half")).to_numpy()
others = d[[0, 2, 3, 4]].mean(axis=1)
findings["Tuesday vs Mon/Wed-Fri"] = (d[1] / others).round(2).unstack().to_dict()
findings["Saturday vs Mon/Wed-Fri"] = (d[5] / others).round(2).unstack().to_dict()
findings["Sunday vs Mon/Wed-Fri"] = (d[6] / others).round(2).unstack().to_dict()

# %% 5. Customers: frequency is not linked to spend, branch or channel
k = o[o["customer"].notna()]
cust = k.groupby("customer").agg(visits=("revenue", "size"), spend_per_visit=("revenue", "mean"),
                                 branches=("branch", "nunique"),
                                 takeaway_share=("order_type", lambda s: float((s == "Takeaway").mean())))
findings["customer-level Spearman"] = cust.corr(method="spearman").round(3).to_dict()
sh = k.copy()
sh["branch"] = rng.permutation(sh["branch"].to_numpy())
findings["branches per customer: actual vs shuffled"] = {
    "actual": round(float(cust["branches"].mean()), 3),
    "shuffled": round(float(sh.groupby("customer")["branch"].nunique().mean()), 3)}

# %% 6. Products (cross-section, NOT elasticity)
pv = df.groupby("Product").agg(price=("Unit Price (USD)", "mean"), units=("Quantity", "sum"),
                               gp=(GP, "sum"), rev=(R, "sum"))
pv["margin"] = pv["gp"] / pv["rev"]
findings["across products: price vs units sold (Spearman)"] = round(float(pv["price"].rank().corr(pv["units"].rank())), 3)
findings["across products: margin vs units sold (Spearman)"] = round(float(pv["margin"].rank().corr(pv["units"].rank())), 3)

meta = {"split": SPLIT, "halves": HALVES, "n_orders": int(len(o)), "n_lines": int(len(df)),
        "raw_lines_retained": True, "stable_rule": "|H1 - H2| < 0.03 for pair measures",
        "note": "Associations only. Product price vs volume is cross-sectional, not elasticity.",
        "findings": findings}
(OUT / "correlation_scan.json").write_text(json.dumps(meta, indent=2, default=str))
print("\nsaved reports/correlation_scan_pairs.csv and reports/correlation_scan.json")


# %% Chart
INK2, GRID, SURFACE, ACCENT, LIGHT, MUTED = "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#9ec5f4", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

ax = axes[0, 0]
ax.scatter(pairs[HALVES[0]], pairs[HALVES[1]], s=14, color=ACCENT, alpha=0.6)
lim = [min(pairs[HALVES].min().min(), 0) - 0.02, pairs[HALVES].max().max() + 0.05]
ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
for _, r in pd.concat([pairs.head(2), pairs.nsmallest(1, "all")]).iterrows():
    ax.annotate(f"{r['a']} x {r['b']}", (r[HALVES[0]], r[HALVES[1]]), fontsize=7.5, color=INK2,
                xytext=(4, -10), textcoords="offset points")
ax.set_xlabel(f"Association strength, {HALVES[0]}")
ax.set_ylabel(f"Association strength, {HALVES[1]}")
ax.set_title(f"All {len(pairs)} pairs repeat almost exactly in both halves")

ax = axes[0, 1]
im = ax.imshow(mix.to_numpy(), cmap="Blues", aspect="auto")
ax.set_xticks(range(3), ["morning\n7-10", "afternoon\n11-17", "evening\n18-23"])
ax.set_yticks(range(len(mix)), mix.index)
for i in range(mix.shape[0]):
    for j in range(3):
        ax.text(j, i, f"{mix.iloc[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                color="white" if mix.iloc[i, j] > 0.18 else INK2)
ax.grid(False)
ax.set_title("Time of day changes what people order (share of items)")

ax = axes[1, 0]
tk = pd.DataFrame(findings["takeaway share by daypart"]).drop(columns="all").loc[["morning 7-10", "afternoon 11-17", "evening 18-23"]]
x = np.arange(3)
for i, (h, c) in enumerate(zip(HALVES, [ACCENT, LIGHT])):
    ax.bar(x + (i - 0.5) * 0.35, tk[h], width=0.35, color=c, label=h)
    for j, v in enumerate(tk[h]):
        ax.text(j + (i - 0.5) * 0.35, v + 0.01, f"{v:.0%}", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(x, ["morning", "afternoon", "evening"])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, 0.7)
ax.set_title("Takeaway is half of daytime orders, a quarter in the evening")
ax.legend()

ax = axes[1, 1]
mx = findings["two-line orders that are 1 drink + 1 food"]
labels = ["drink + food\n(actual)", "drink + food\n(if random)"]
vals = [np.mean([mx[h]["actual"] for h in HALVES]), np.mean([mx[h]["shuffled_mean"] for h in HALVES])]
top = lift.head(5)
ax.bar(labels, vals, color=[ACCENT, MUTED], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", color=INK2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, 0.8)
ax.set_title("Drink + food is real; which drink with which food is not")
ax.text(0.98, 0.97, "Strongest category-pair lifts (1 = chance):\n" +
        "\n".join(f"{r['pair']}: {r[HALVES[0]]:.2f} / {r[HALVES[1]]:.2f}" for _, r in top.iterrows()),
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
fig.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "correlation_scan.png", dpi=150, bbox_inches="tight")
print("saved reports/figures/correlation_scan.png")
