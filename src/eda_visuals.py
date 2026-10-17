"""
Stories Café EDA: visualize the key findings.

Run the whole script from the repo root:
    python src/eda_visuals.py          # opens each chart in a window
    python src/eda_visuals.py --save   # also saves every chart as a PNG to reports/figures/

In VS Code you can also run it cell by cell: each "# %%" line starts a cell,
and "Run Cell" shows the chart in the Interactive Window.

Requires: pandas, numpy, matplotlib
"""

# %% Setup
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Repo root = first folder (from here upward) that contains data/stories_transactions.csv
_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
DATA = ROOT / "data" / "stories_transactions.csv"
FIG_DIR = ROOT / "reports" / "figures"
SAVE = "--save" in sys.argv

# Each branch keeps the same color in every chart.
BRANCH_COLORS = {"Harbour": "#2a78d6", "Garden": "#eb6834", "Uptown": "#1baf7a"}
BRANCHES = list(BRANCH_COLORS)
ACCENT, MUTED = "#2a78d6", "#b9b8b3"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2,
})
USD = mtick.StrMethodFormatter("${x:,.0f}")
PCT = mtick.PercentFormatter(1.0, decimals=0)
R, GP = "Line Revenue (USD)", "Line Gross Profit (USD)"


def finish(fig, name):
    """Save the figure (with --save) and show it."""
    fig.tight_layout()
    if SAVE:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"  saved reports/figures/{name}.png")
    plt.show()


# %% Load and clean
raw = pd.read_csv(DATA)
df = raw.drop_duplicates().copy()  # 132 identical lines within the same order
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df["dow"] = pd.Categorical(df["Date"].dt.day_name(),
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)
df["daypart"] = pd.cut(df["hour"], [0, 10, 17, 24], include_lowest=True,
                       labels=["Morning (7-10)", "Afternoon (11-17)", "Evening (18-23)"])

# One row per order (Transaction ID = order, a CSV row = one product line)
orders = df.groupby("Transaction ID").agg(
    date=("Date", "first"), hour=("hour", "first"), dow=("dow", "first"), branch=("Branch", "first"),
    order_type=("Order Type", "first"), customer=("Customer ID", "first"),
    revenue=(R, "sum"), profit=(GP, "sum"), items=("Quantity", "sum"), lines=("Product", "size"),
    discount=("Discount Rate", "max"),
)

print("=== Data quality ===")
print(f"Rows: {len(raw):,} | duplicates dropped: {raw.duplicated().sum()} | orders: {len(orders):,}")
print(f"Missing Customer ID: {raw['Customer ID'].isna().mean():.1%}")
print(f"Period: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
print("\n=== Headline KPIs ===")
print(f"Revenue ${df[R].sum():,.0f} | Gross profit ${df[GP].sum():,.0f} | Margin {df[GP].sum() / df[R].sum():.1%}")
print(f"Average order value ${orders['revenue'].mean():.2f} | Items per order {orders['items'].mean():.2f}")


# %% Finding 1: Garden's cost base
unit = df.groupby(["Product", "Branch"])[["Unit Price (USD)", "Unit Cost (USD)"]].first().unstack()
price_idx = unit["Unit Price (USD)"].div(unit["Unit Price (USD)"]["Harbour"], axis=0).mean()
cost_idx = unit["Unit Cost (USD)"].div(unit["Unit Cost (USD)"]["Harbour"], axis=0).mean()
branch_margin = df.groupby("Branch")[GP].sum() / df.groupby("Branch")[R].sum()

g = df[df["Branch"] == "Garden"]
gap = (g["Quantity"] * g["Unit Cost (USD)"]).sum() - (
    g["Quantity"] * g["Product"].map(unit["Unit Cost (USD)"]["Harbour"])).sum()

print("\n=== Finding 1: Garden's cost base ===")
print(pd.DataFrame({"price vs Harbour": price_idx, "unit cost vs Harbour": cost_idx,
                    "gross margin": branch_margin}).loc[BRANCHES].map("{:.0%}".format))
print(f"Extra gross profit if Garden had Harbour's unit costs (6 months): ${gap:,.0f}")

cat_margin = (df.pivot_table(index="Category", columns="Branch", values=GP, aggfunc="sum")
              / df.pivot_table(index="Category", columns="Branch", values=R, aggfunc="sum"))[BRANCHES]
cat_margin = cat_margin.sort_values("Harbour")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.8]})
ax = axes[0]
x = np.arange(len(BRANCHES))
ax.bar(x, branch_margin[BRANCHES], color=[BRANCH_COLORS[b] for b in BRANCHES], width=0.6)
for i, b in enumerate(BRANCHES):
    ax.text(i, branch_margin[b] + 0.01, f"{branch_margin[b]:.0%}", ha="center", color=INK2)
    ax.text(i, 0.03, f"cost {cost_idx[b]:.0%}\nprice {price_idx[b]:.0%}", ha="center",
            color="white", fontsize=8.5)
ax.set_xticks(x, BRANCHES)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, 0.75)
ax.set_title("Gross margin by branch")
ax.text(0, -0.13, "cost / price shown relative to Harbour", transform=ax.transAxes, color=INK2, fontsize=8.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
y = np.arange(len(cat_margin))
ax.hlines(y, cat_margin.min(axis=1), cat_margin.max(axis=1), color=GRID, lw=2, zorder=2)
for b in BRANCHES:
    ax.scatter(cat_margin[b], y, s=60, color=BRANCH_COLORS[b], label=b, zorder=3,
               edgecolors=SURFACE, linewidths=2)
ax.set_yticks(y, cat_margin.index)
ax.xaxis.set_major_formatter(PCT)
ax.set_xlabel("Gross margin")
ax.set_title("Garden trails in every category")
ax.legend(loc="upper left", ncols=3)
ax.grid(axis="y", visible=False)
finish(fig, "1_garden_cost_base")


# %% Finding 2: Growth comes from more orders, not bigger baskets
weekly = orders.set_index("date").groupby("branch").resample("W-SUN")["revenue"].agg(["size", "mean"])
weekly_orders = weekly["size"].unstack(0)[BRANCHES].iloc[1:-1]  # drop partial first/last week
weekly_aov = weekly["mean"].unstack(0)[BRANCHES].iloc[1:-1]

print("\n=== Finding 2: Growth ===")
monthly = orders.groupby(orders["date"].dt.to_period("M")).agg(orders=("revenue", "size"), aov=("revenue", "mean"))
print(monthly.round(2))
print(f"Single-line orders: {(orders['lines'] == 1).mean():.0%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, data, title, fmt in [(axes[0], weekly_orders, "Weekly orders: up ~50% since January", None),
                             (axes[1], weekly_aov, "Average order value: flat at ~$9", "${x:,.2f}")]:
    for b in BRANCHES:
        ax.plot(data.index, data[b], color=BRANCH_COLORS[b], label=b)
        if not fmt:  # AOV lines overlap at the end, so the legend identifies them there
            ax.annotate(b, (data.index[-1], data[b].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
    ax.set_title(title)
    ax.margins(x=0.1)
    ax.legend(loc="upper left", ncols=3)
    if fmt:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(fmt))
        ax.set_ylim(0, 12)
    else:
        ax.set_ylim(0)
fig.autofmt_xdate()
finish(fig, "2_growth_orders_vs_aov")


# %% Finding 3: Evening is the core business
ot = pd.crosstab(orders["hour"], orders["order_type"], normalize="index")
hourly = orders.groupby("hour").size() / df["Date"].nunique()
mix = pd.crosstab(df["daypart"], df["Category"], values=df["Quantity"], aggfunc="sum", normalize="index")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
ax = axes[0]
ax.bar(hourly.index, hourly.values, color=[ACCENT if h == hourly.idxmax() else MUTED for h in hourly.index], width=0.7)
ax.set_xticks(hourly.index[::2])
ax.set_xlabel("Hour of day")
ax.set_title("Orders per day by hour (peak 21:00)")
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.plot(ot.index, ot["Dine-in"], color=ACCENT, marker="o", ms=4)
ax.axvspan(17.5, 23.5, color=GRID, alpha=0.5, lw=0)
ax.text(20.5, 0.9, "Evening: mostly dine-in", ha="center", color=INK2)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, 1)
ax.set_xticks(ot.index[::2])
ax.set_xlabel("Hour of day")
ax.set_title("Share of orders that are dine-in")

ax = axes[2]
show = ["Bakery", "Hot Coffee", "Cold Coffee", "Matcha & Tea", "Desserts"]
mix[show].T.plot.barh(ax=ax, color=[SEQ[2], SEQ[4], SEQ[6]], width=0.75, edgecolor=SURFACE, linewidth=1)
ax.xaxis.set_major_formatter(PCT)
ax.set_title("Unit mix by daypart")
ax.set_ylabel("")
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
finish(fig, "3_evening_core")


# %% Finding 4: Tuesday boom, weak weekends
days_per_dow = df.groupby("dow", observed=True)["Date"].nunique()
dow_avg = orders.groupby("dow", observed=True).size() / days_per_dow
heat = orders.pivot_table(index="dow", columns="hour", values="revenue", aggfunc="size", observed=True)
heat = heat.div(days_per_dow, axis=0)

print("\n=== Finding 4: Day of week ===")
print(dow_avg.round(0))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
ax = axes[0]
colors = [ACCENT if d == dow_avg.idxmax() else MUTED for d in dow_avg.index]
ax.bar(dow_avg.index.astype(str).str[:3], dow_avg.values, color=colors, width=0.6)
for i, v in enumerate(dow_avg.values):
    ax.text(i, v + 3, f"{v:.0f}", ha="center", color=INK2)
ax.set_title("Average orders per day, by weekday")
ax.grid(axis="x", visible=False)

ax = axes[1]
im = ax.imshow(heat.values, aspect="auto", cmap=LinearSegmentedColormap.from_list("seq", SEQ))
ax.set_xticks(range(heat.shape[1]), [f"{h}h" for h in heat.columns], fontsize=8)
ax.set_yticks(range(heat.shape[0]), heat.index.astype(str))
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("Avg orders per hour", color=INK2)
cb.outline.set_visible(False)
ax.set_title("Demand heatmap: weekday x hour")
finish(fig, "4_weekday_hour")


# %% Finding 5: Food margins are thin
cat = df.groupby("Category").agg(revenue=(R, "sum"), profit=(GP, "sum"))
cat["margin"] = cat["profit"] / cat["revenue"]
cat = cat.sort_values("margin")
prod = df.groupby(["Product", "Category"]).agg(revenue=(R, "sum"), profit=(GP, "sum")).reset_index()
prod["margin"] = prod["profit"] / prod["revenue"]
overall = df[GP].sum() / df[R].sum()

print("\n=== Finding 5: Category margins ===")
print(cat.sort_values("profit", ascending=False).round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axes[0]
food = {"Sandwiches", "Desserts", "Meals", "Breakfast", "Bakery"}
ax.barh(cat.index, cat["margin"], color=["#eb6834" if c in food else ACCENT for c in cat.index], height=0.6)
for i, v in enumerate(cat["margin"]):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center", color=INK2)
ax.axvline(overall, color=MUTED, ls="--", lw=1)
ax.xaxis.set_major_formatter(PCT)
ax.set_xlim(0, 0.8)
ax.set_title("Gross margin by category (orange = food)")
ax.grid(axis="y", visible=False)

ax = axes[1]
ax.scatter(prod["revenue"], prod["margin"], s=40, color=ACCENT, edgecolors=SURFACE, linewidths=1.5, zorder=3)
ax.axvline(prod["revenue"].median(), color=MUTED, lw=1, ls="--")
ax.axhline(overall, color=MUTED, lw=1, ls="--")
label = pd.concat([prod.nlargest(5, "revenue"), prod.nsmallest(4, "margin"),
                   prod.nlargest(3, "margin")]).drop_duplicates("Product")
nudge = {"Turkey & Cheese Sandwich": (-5, -12, "right"), "Mint Tea": (6, -3, "left")}
for _, r in label.iterrows():
    dx, dy, ha = nudge.get(r["Product"], (5, 4, "left"))
    ax.annotate(r["Product"], (r["revenue"], r["margin"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, color=INK2, ha=ha)
for xq, yq, t in [(0.98, 0.98, "Stars"), (0.98, 0.02, "High volume, low margin: reprice / cut cost"),
                  (0.02, 0.98, "Niche, high margin: promote"), (0.02, 0.02, "Review")]:
    ax.text(xq, yq, t, transform=ax.transAxes, ha="right" if xq > 0.5 else "left",
            va="top" if yq > 0.5 else "bottom", fontsize=8.5, color=INK2, style="italic")
ax.set_ylim(0.40, 0.87)
ax.xaxis.set_major_formatter(USD)
ax.yaxis.set_major_formatter(PCT)
ax.set_xlabel("Revenue (6 months)")
ax.set_ylabel("Gross margin")
ax.set_title("Product portfolio: revenue vs margin")
finish(fig, "5_margins")


# %% Finding 6: The 15% promotion did not pay
pm = df[df["Date"].dt.month.isin([3, 5])].copy()
pm["promo_week"] = pm["Date"].between("2026-03-12", "2026-03-19") | pm["Date"].between("2026-05-12", "2026-05-19")
promo_units = (pm[pm["Category"].isin(["Cold Coffee", "Desserts"])]
               .groupby(["Branch", "promo_week", "Date"])["Quantity"].sum()
               .groupby(["Branch", "promo_week"]).mean().unstack())
promo_profit = (pm.groupby(["Branch", "promo_week", "Date"])[GP].sum()
                .groupby(["Branch", "promo_week"]).mean().unstack())
lift = pd.DataFrame({"Cold Coffee + Desserts units/day": promo_units[True] / promo_units[False] - 1,
                     "Total gross profit/day": promo_profit[True] / promo_profit[False] - 1}).loc[BRANCHES]

print("\n=== Finding 6: 15% promo (Harbour only, Mar 12-19 and May 12-19) ===")
print(lift.map("{:+.1%}".format))

fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.38
x = np.arange(len(lift.columns))
for i, b in enumerate(BRANCHES):
    vals = lift.loc[b].values
    bars = ax.bar(x + (i - 1) * w * 0.7, vals, width=w * 0.65, color=BRANCH_COLORS[b],
                  label=b + (" (promo)" if b == "Harbour" else " (no promo)"))
    for xi, v in zip(x + (i - 1) * w * 0.7, vals):
        ax.text(xi, v + (0.006 if v >= 0 else -0.006), f"{v:+.0%}", ha="center",
                va="bottom" if v >= 0 else "top", color=INK2, fontsize=9)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x, lift.columns)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylim(-0.2, 0.1)
ax.set_title("15% promo weeks vs rest of the month: no volume lift at Harbour")
ax.legend(loc="lower left", ncols=3)
ax.grid(axis="x", visible=False)
finish(fig, "6_promo_effect")


# %% Finding 7: Customers are loyal to the brand, not one branch
known = orders.dropna(subset=["customer"])
cust = known.groupby("customer").agg(orders=("revenue", "size"), branches=("branch", "nunique"),
                                     first=("date", "min"))
print("\n=== Finding 7: Customers ===")
print(f"Identified customers: {len(cust):,} ({len(known) / len(orders):.0%} of orders)")
print(f"Median visits: {cust['orders'].median():.0f} | used 2+ branches: {(cust['branches'] >= 2).mean():.0%}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
vc = cust["orders"].value_counts().sort_index()
axes[0].bar(vc.index, vc.values, color=ACCENT, width=0.7)
axes[0].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
axes[0].set_xlabel("Orders in 6 months")
axes[0].set_title("Visits per identified customer")

bc = cust["branches"].value_counts().sort_index()
axes[1].bar(bc.index.astype(str), bc.values, color=[MUTED, ACCENT, ACCENT], width=0.6)
for i, v in enumerate(bc.values):
    axes[1].text(i, v + 25, f"{v / bc.sum():.0%}", ha="center", color=INK2)
axes[1].set_xlabel("Number of branches visited")
axes[1].set_title("Most customers use 2-3 branches")

newc = cust["first"].dt.to_period("M").value_counts().sort_index()
axes[2].bar(newc.index.astype(str), newc.values, color=ACCENT, width=0.6)
for i, v in enumerate(newc.values):
    axes[2].text(i, v + 20, f"{v:,}", ha="center", color=INK2)
axes[2].set_title("New identified customers by month")
for ax in axes:
    ax.grid(axis="x", visible=False)
finish(fig, "7_customers")


# %% Finding 8: Cross-sell (what is bought together)
basket_prod = df.groupby("Transaction ID")["Product"].apply(lambda s: tuple(sorted(set(s))))
pairs = Counter(p for t in basket_prod for p in combinations(t, 2)).most_common(12)
pairs_df = pd.DataFrame([(f"{a} + {b}", n) for (a, b), n in pairs], columns=["pair", "orders"])

lines = orders["lines"].value_counts(normalize=True).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
ax = axes[0]
ax.bar([f"{i} product{'s' if i > 1 else ''}" for i in lines.index], lines.values,
       color=[ACCENT, MUTED, MUTED], width=0.6)
for i, v in enumerate(lines.values):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", color=INK2)
ax.yaxis.set_major_formatter(PCT)
ax.set_title("Distinct products per order")
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.barh(pairs_df["pair"][::-1], pairs_df["orders"][::-1], color=ACCENT, height=0.6)
for i, v in enumerate(pairs_df["orders"][::-1]):
    ax.text(v + 1, i, str(v), va="center", color=INK2, fontsize=9)
ax.set_title("Most common product pairs in the same order")
ax.grid(axis="y", visible=False)
finish(fig, "8_cross_sell")

print("\nDone.")
