"""
Stories Café: charts behind the pitch.

Pitch: demand is a habit, not a reaction. The same loyal customers come back more often,
on the same days and hours, whatever the price, weather or news. That makes the price
list (not promotions) the lever: fix Garden's and Uptown's prices and Garden's cost base,
stop untargeted discounts, and turn single-item orders into two-item orders.

Run from the repo root:
    python src/pitch_visuals.py          # opens each chart in a window
    python src/pitch_visuals.py --save   # also saves every chart to reports/figures/pitch_*.png

External data: data/external/beirut_weather_2026H1.json (Open-Meteo archive API, Beirut,
daily). If the file is missing it is downloaded once.

Requires: pandas, numpy, matplotlib
"""

# %% Setup
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
DATA = ROOT / "data" / "stories_transactions.csv"
WEATHER = ROOT / "data" / "external" / "beirut_weather_2026H1.json"
WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=33.89&longitude=35.50"
               "&start_date=2026-01-01&end_date=2026-06-30"
               "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
               "&timezone=Asia%2FBeirut")
FIG_DIR = ROOT / "reports" / "figures"
SAVE = "--save" in sys.argv

BRANCH_COLORS = {"Harbour": "#2a78d6", "Garden": "#eb6834", "Uptown": "#1baf7a"}
BRANCHES = list(BRANCH_COLORS)
ACCENT, WARN, MUTED = "#2a78d6", "#d64545", "#b9b8b3"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

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
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"  saved reports/figures/{name}.png")
    plt.show()


# %% Load and clean
df = pd.read_csv(DATA).drop_duplicates().copy()
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df["month"] = df["Date"].dt.month

orders = df.groupby("Transaction ID").agg(
    date=("Date", "first"), hour=("hour", "first"), branch=("Branch", "first"),
    order_type=("Order Type", "first"), customer=("Customer ID", "first"),
    revenue=(R, "sum"), profit=(GP, "sum"), items=("Quantity", "sum"), lines=("Product", "size"),
)
orders["month"] = orders["date"].dt.month

if not WEATHER.exists():
    WEATHER.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WEATHER_URL, WEATHER)
weather = pd.DataFrame(json.loads(WEATHER.read_text())["daily"])
weather["date"] = pd.to_datetime(weather.pop("time"))
weather = weather.set_index("date")

# Dated events in Lebanon (sources: Wikipedia "2026 in Lebanon" and
# "Timeline of the 2026 Lebanon war"; Islamic holiday dates vary by a day across sources).
EVENTS = {
    "2026-03-02": "War starts",
    "2026-03-12": "Beirut strike",
    "2026-03-20": "Eid al-Fitr",
    "2026-04-08": "Deadliest day\n(~357 killed)",
    "2026-04-17": "Truce",
    "2026-05-27": "Eid al-Adha",
}
RAMADAN = ("2026-02-18", "2026-03-19")
WAR = ("2026-03-02", "2026-04-16")


# %% Pitch 1: Demand is a habit. Day-of-week + a straight trend explain almost all of it.
daily = orders.groupby("date").agg(orders=("revenue", "size"),
                                   takeaway=("order_type", lambda s: (s == "Takeaway").mean()))
daily = daily.join(weather)
X = pd.get_dummies(daily.index.dayofweek, prefix="dow").astype(float).set_index(daily.index)
X["trend"] = np.arange(len(daily))
beta, *_ = np.linalg.lstsq(X.values, daily["orders"].values, rcond=None)
daily["expected"] = X.values @ beta
daily["residual"] = daily["orders"] / daily["expected"] - 1
r2 = 1 - ((daily["orders"] - daily["expected"]) ** 2).sum() / ((daily["orders"] - daily["orders"].mean()) ** 2).sum()

print("=== Pitch 1: demand is a habit ===")
print(f"Day-of-week + linear trend explain R² = {r2:.0%} of daily orders")
for col, label in [("temperature_2m_max", "max temperature"), ("precipitation_sum", "rain"),
                   ("wind_speed_10m_max", "wind")]:
    print(f"  correlation of leftover demand with {label}: {daily['residual'].corr(daily[col]):+.2f}")
for day, label in EVENTS.items():
    r = daily.loc[day]
    print(f"  {day} {label.replace(chr(10), ' '):<28} {int(r.orders):>4} orders, {r.residual:+.0%} vs expected")

fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
for (a, b), color, label in [(RAMADAN, "#f3e7c9", "Ramadan"), (WAR, "#f6d5d5", "War, before truce")]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=color, alpha=0.6, lw=0, label=label)
ax.plot(daily.index, daily["orders"], color=MUTED, lw=1.2, label="Actual orders per day")
ax.plot(daily.index, daily["expected"], color=ACCENT, lw=1.6,
        label=f"Predicted from weekday + trend only (R² = {r2:.0%})")
ymax = daily["orders"].max()
for i, (day, label) in enumerate(EVENTS.items()):
    d = pd.Timestamp(day)
    ax.axvline(d, color=INK2, lw=0.8, ls=":")
    ax.text(d, ymax * (1.02 + 0.1 * (i % 2)), label, fontsize=8, color=INK2, ha="center", va="bottom",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
ax.set_ylim(0, ymax * 1.32)
ax.set_ylabel("Orders per day (all branches)")
ax.set_title("Sales follow the calendar, not the news: a war, two Eids and Ramadan barely register")
ax.legend(loc="lower right", ncols=2, fontsize=8.5)
ax.grid(axis="x", visible=False)

ax = axes[1]
ax.bar(daily.index, daily["precipitation_sum"], color="#9ec5f4", width=1, label="Rain (mm)")
ax.set_ylabel("Rain (mm)")
ax2 = ax.twinx()
ax2.plot(daily.index, daily["temperature_2m_max"], color=BRANCH_COLORS["Garden"], lw=1.4, label="Max temp (°C)")
ax2.set_ylabel("Max temp (°C)")
ax2.spines["right"].set_visible(True)
ax2.grid(False)
ax.set_title(f"Beirut weather: corr. with leftover demand is "
             f"{daily['residual'].corr(daily['temperature_2m_max']):+.2f} (temp) and "
             f"{daily['residual'].corr(daily['precipitation_sum']):+.2f} (rain)", fontsize=10.5)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
ax.grid(axis="x", visible=False)
finish(fig, "pitch_1_demand_is_habit")


# %% Pitch 1b: Nothing in the mix moves either (summer heat, war, Ramadan)
cold = (df["Category"].isin(["Cold Coffee", "Cold Drinks"])
        | df["Product"].str.contains("Iced|Frozen|Frappe"))
monthly_mix = pd.DataFrame({
    "Max temp (°C)": weather.groupby(weather.index.month)["temperature_2m_max"].mean(),
    "Cold-drink share of items": df[cold].groupby("month")["Quantity"].sum() / df.groupby("month")["Quantity"].sum(),
    "Takeaway share of orders": orders.groupby("month")["order_type"].apply(lambda s: (s == "Takeaway").mean()),
})
ram = orders["date"].between(*RAMADAN)
pre = orders["date"] < RAMADAN[0]
print("\n=== Pitch 1b: mix does not react ===")
print(monthly_mix.round(3))
print(f"Daytime (11-17h) share of orders: Ramadan {orders.loc[ram, 'hour'].between(11, 17).mean():.1%}"
      f" vs before Ramadan {orders.loc[pre, 'hour'].between(11, 17).mean():.1%}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
ax = axes[0]
ax.bar(months, monthly_mix["Max temp (°C)"], color=BRANCH_COLORS["Garden"], width=0.6)
ax.set_title("Beirut got 11°C hotter...")
ax.set_ylabel("Avg daily max (°C)")
ax.grid(axis="x", visible=False)
ax = axes[1]
ax.plot(months, monthly_mix["Cold-drink share of items"], marker="o", color=ACCENT)
ax.set_ylim(0, 0.5)
ax.yaxis.set_major_formatter(PCT)
ax.set_title("...but iced drinks stayed at ~31%")
ax = axes[2]
ax.plot(months, monthly_mix["Takeaway share of orders"], marker="o", color=ACCENT)
ax.axvspan(1.5, 3.5, color="#f6d5d5", alpha=0.6, lw=0)
ax.text(2.5, 0.47, "war", ha="center", color=WARN, fontsize=9)
ax.set_ylim(0, 0.5)
ax.yaxis.set_major_formatter(PCT)
ax.set_title("...and takeaway share ignored the war")
finish(fig, "pitch_1b_mix_does_not_react")


# %% Pitch 2: Same loyal customers, coming back more often (acquisition has stopped)
known = orders[orders["customer"].notna()].copy()
first_seen = known.groupby("customer")["date"].min()
cust_m = known.groupby("month").agg(orders=("revenue", "size"), active=("customer", "nunique"))
cust_m["new"] = first_seen.dt.month.value_counts().sort_index()
cust_m["returning"] = cust_m["active"] - cust_m["new"]
cust_m["visits_per_active"] = cust_m["orders"] / cust_m["active"]
early = set(known.loc[known["month"] <= 2, "customer"])
late = set(known.loc[known["month"] >= 5, "customer"])

print("\n=== Pitch 2: loyal base ===")
print(cust_m.round(2))
print(f"Identified customers: {known['customer'].nunique():,} | "
      f"Jan-Feb customers still active in May-Jun: {len(early & late) / len(early):.0%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.bar(months, cust_m["returning"], color=ACCENT, width=0.6, label="Returning customers")
ax.bar(months, cust_m["new"], bottom=cust_m["returning"], color=BRANCH_COLORS["Garden"], width=0.6,
       label="New customers")
for i, n in enumerate(cust_m["new"]):
    ax.text(i, cust_m["active"].iloc[i] + 30, f"+{n:,} new", ha="center", fontsize=8.5, color=INK2)
ax.set_ylabel("Active identified customers")
ax.set_title("New customers dried up: 1,979 in Jan, 20 in Jun")
ax.legend(loc="lower right")
ax.grid(axis="x", visible=False)
ax = axes[1]
ax.plot(months, cust_m["visits_per_active"], marker="o", color=ACCENT)
for i, v in enumerate(cust_m["visits_per_active"]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
ax.set_ylim(1, 2)
ax.set_ylabel("Orders per active customer per month")
ax.set_title(f"Growth = the same people visiting more ({len(early & late) / len(early):.0%} retained)")
finish(fig, "pitch_2_loyal_base")


# %% Pitch 3: The price list is backwards (Garden pays the most, charges less)
unit = df.groupby(["Product", "Branch"])[["Unit Price (USD)", "Unit Cost (USD)"]].first().unstack()
price_idx = unit["Unit Price (USD)"].div(unit["Unit Price (USD)"]["Harbour"], axis=0)
cost_idx = unit["Unit Cost (USD)"].div(unit["Unit Cost (USD)"]["Harbour"], axis=0)
branch_rev = df.groupby("Branch")[R].sum()
branch_gp = df.groupby("Branch")[GP].sum()
branch_margin = branch_gp / branch_rev

print("\n=== Pitch 3: price list ===")
print(pd.DataFrame({"price vs Harbour (min-max)": price_idx.agg(["min", "max"]).T.apply(lambda r: f"{r['min']:.1%}-{r['max']:.1%}", axis=1),
                    "cost vs Harbour (min-max)": cost_idx.agg(["min", "max"]).T.apply(lambda r: f"{r['min']:.1%}-{r['max']:.1%}", axis=1),
                    "gross margin": branch_margin.map("{:.0%}".format)}).loc[BRANCHES])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for b in BRANCHES:
    ax.scatter(price_idx[b], cost_idx[b], s=40, color=BRANCH_COLORS[b], alpha=0.8, label=b,
               edgecolors=SURFACE, linewidths=1)
ax.axhline(1, color=GRID, lw=1)
ax.axvline(1, color=GRID, lw=1)
ax.annotate("Garden: every product costs ~43% more\nand sells for ~5% less than at Harbour",
            (price_idx["Garden"].mean(), cost_idx["Garden"].mean()), xytext=(0.965, 1.22),
            fontsize=9, color=BRANCH_COLORS["Garden"], ha="left",
            arrowprops=dict(arrowstyle="->", color=BRANCH_COLORS["Garden"]))
ax.annotate("Uptown: ~9% cheaper,\ncosts only ~4% lower", (price_idx["Uptown"].mean(), cost_idx["Uptown"].mean()),
            xytext=(0.925, 0.85), fontsize=9, color=BRANCH_COLORS["Uptown"],
            arrowprops=dict(arrowstyle="->", color=BRANCH_COLORS["Uptown"]))
ax.xaxis.set_major_formatter(PCT)
ax.yaxis.set_major_formatter(PCT)
ax.set_xlim(0.88, 1.03)
ax.set_ylim(0.8, 1.55)
ax.set_xlabel("Unit price relative to Harbour (each dot = one of 45 products)")
ax.set_ylabel("Unit cost relative to Harbour")
ax.set_title("Flat multipliers on every product = a price-list setting")
ax.legend(loc="upper right")
ax = axes[1]
x = np.arange(len(BRANCHES))
ax.bar(x, branch_margin[BRANCHES], color=[BRANCH_COLORS[b] for b in BRANCHES], width=0.6)
for i, b in enumerate(BRANCHES):
    ax.text(i, branch_margin[b] + 0.01, f"{branch_margin[b]:.0%}", ha="center", color=INK2)
ax.set_xticks(x, BRANCHES)
ax.yaxis.set_major_formatter(PCT)
ax.set_ylim(0, 0.8)
ax.set_title("Gross margin by branch")
ax.grid(axis="x", visible=False)
finish(fig, "pitch_3_price_list")


# %% Pitch 4: A 15% price cut bought zero extra volume (Harbour promo vs other branches)
PROMO_STARTS = ["2026-03-12", "2026-05-12"]  # 8 days each, Harbour only, Cold Coffee + Desserts at 15%
PROMO_CATS = ["Cold Coffee", "Desserts"]


def window(d):
    for s in map(pd.Timestamp, PROMO_STARTS):
        if s <= d <= s + pd.Timedelta(days=7):
            return "During promo"
        if s - pd.Timedelta(days=14) <= d < s:
            return "2 weeks before"
        if s + pd.Timedelta(days=8) <= d <= s + pd.Timedelta(days=21):
            return "2 weeks after"
    return None


df["window"] = df["Date"].map(window)
w = df[df["window"].notna() & df["Category"].isin(PROMO_CATS)]
promo = w.groupby(["Branch", "window"]).agg(qty=("Quantity", "sum"), gp=(GP, "sum"), days=("Date", "nunique"))
promo["qty/day"] = promo["qty"] / promo["days"]
promo["gp/day"] = promo["gp"] / promo["days"]
WINDOWS = ["2 weeks before", "During promo", "2 weeks after"]
pq = promo["qty/day"].unstack()[WINDOWS]
pg = promo["gp/day"].unstack()[WINDOWS]
promo_lines = df[df["Discount Rate"] == 0.15]
promo_cost = (promo_lines["Quantity"] * promo_lines["Unit Price (USD)"] * 0.15).sum()

print("\n=== Pitch 4: promo ===")
print("Promo-category units/day:\n", pq.round(1))
print("Promo-category gross profit/day:\n", pg.round(0))
print(f"Discount given away over 16 promo days: ${promo_cost:,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, table, title, fmt in [(axes[0], pq / pq["2 weeks before"].values[:, None], "Units sold: no lift at Harbour", PCT),
                              (axes[1], pg / pg["2 weeks before"].values[:, None], "Gross profit: Harbour drops ~23%", PCT)]:
    for b in BRANCHES:
        lw, alpha = (3, 1) if b == "Harbour" else (1.5, 0.6)
        ax.plot(WINDOWS, table.loc[b], marker="o", lw=lw, alpha=alpha, color=BRANCH_COLORS[b],
                label=b + (" (15% off)" if b == "Harbour" else " (no promo)"))
    ax.axhline(1, color=GRID, lw=1)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(0.6, 1.3)
    ax.set_ylabel("Indexed to 2 weeks before")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8.5)
fig.suptitle("Cold Coffee + Desserts, Harbour 15% promo (12-19 Mar and 12-19 May)", x=0.01, ha="left",
             fontsize=10, color=INK2)
finish(fig, "pitch_4_promo_no_lift")


# %% Pitch 5: Basket upside. Half the orders are a single item. Customers do NOT have a personal
# "usual" (their top-product share matches random ordering), so a simple till add-on beats
# personalised offers.
single = (orders["items"] == 1).mean()
by_items = orders["items"].clip(upper=4).value_counts(normalize=True).sort_index()
kl = df[df["Customer ID"].notna()]
lines_per_c = kl.groupby("Customer ID").size()
top_share = kl.groupby(["Customer ID", "Product"]).size().groupby(level=0).max() / lines_per_c
top_share = top_share[lines_per_c >= 10]
# What a customer's top share would be if they picked products at the overall menu mix
rng = np.random.default_rng(0)
mix_p = kl["Product"].value_counts(normalize=True).values
sim = np.array([np.bincount(rng.choice(len(mix_p), n, p=mix_p), minlength=len(mix_p)).max() / n
                for n in lines_per_c[lines_per_c >= 10].values])
pair = df[df["Category"].isin(["Bakery", "Desserts"])]
attach_gp = (pair[GP].sum() / pair["Quantity"].sum())

print("\n=== Pitch 5: basket ===")
print(f"Single-item orders: {single:.0%} | gross profit per bakery/dessert unit: ${attach_gp:.2f}")
print(f"Customers' top-product share: median {top_share.median():.1%} vs {np.median(sim):.1%} if random")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
labels = ["1 item", "2 items", "3 items", "4+ items"]
ax.bar(labels, by_items.values, color=[WARN] + [MUTED] * 3, width=0.6)
for i, v in enumerate(by_items.values):
    ax.text(i, v + 0.01, f"{v:.0%}", ha="center", color=INK2)
ax.yaxis.set_major_formatter(PCT)
ax.set_title(f"{single:.0%} of orders are a single item")
ax.grid(axis="x", visible=False)
ax = axes[1]
bins = np.linspace(0, 0.5, 26)
ax.hist(sim, bins=bins, color=MUTED, alpha=0.8, label="If customers ordered at random")
ax.hist(top_share, bins=bins, color=ACCENT, alpha=0.8, label="Actual customers")
ax.xaxis.set_major_formatter(PCT)
ax.set_xlabel("Share of a customer's items that are their #1 product (customers with 10+ items)")
ax.set_ylabel("Customers")
ax.set_title("No personal 'usual': offer one add-on at the till, not personalised deals")
ax.legend()
finish(fig, "pitch_5_basket_upside")


# %% Pitch 6: What it is worth (6 months observed, x2 for a year) and how much volume we can lose
g = df[df["Branch"] == "Garden"]
garden_cost_gap = (g["Quantity"] * g["Unit Cost (USD)"]).sum() - (
    g["Quantity"] * g["Product"].map(unit["Unit Cost (USD)"]["Harbour"])).sum()
price_up = {b: 1 / price_idx[b].mean() - 1 for b in ["Garden", "Uptown"]}
price_gain = {b: branch_rev[b] * price_up[b] for b in price_up}
# Break-even volume loss for a price rise p at margin m: p / (m + p)
breakeven = {b: price_up[b] / (branch_margin[b] + price_up[b]) for b in price_up}
disc = df[df["Discount Rate"] > 0]
discount_cost = (disc["Quantity"] * disc["Unit Price (USD)"] * disc["Discount Rate"]).sum()
TAKE_UP = 0.10  # assumption: 1 in 10 single-item orders accepts a pastry add-on
basket_gain = (orders["items"] == 1).sum() * TAKE_UP * attach_gp

levers = pd.Series({
    "Garden costs to\nHarbour level": garden_cost_gap,
    "Garden prices +5%\nto Harbour level": price_gain["Garden"],
    "Uptown prices +9%\nto Harbour level": price_gain["Uptown"],
    "Stop untargeted\ndiscounts": discount_cost,
    f"Pastry add-on\n({TAKE_UP:.0%} take-up)": basket_gain,
}) * 2  # annualise the 6 observed months
kind = ["measured gap", "estimate, no volume loss", "estimate, no volume loss", "measured cost", "estimate, assumed take-up"]

print("\n=== Pitch 6: value (per year, gross profit) ===")
for (k, v), t in zip(levers.items(), kind):
    print(f"  {k.replace(chr(10), ' '):<38} ${v:>8,.0f}  ({t})")
print(f"  Total ${levers.sum():,.0f}")
for b in breakeven:
    print(f"  {b} price rise pays off unless volume falls more than {breakeven[b]:.0%}")

fig, ax = plt.subplots(figsize=(13, 4.8))
cum = levers.cumsum()
starts = cum - levers
colors = [ACCENT if "measured" in t else "#9ec5f4" for t in kind]
ax.bar(levers.index, levers.values, bottom=starts.values, color=colors, width=0.6)
for i, (v, top) in enumerate(zip(levers.values, cum.values)):
    ax.text(i, top + levers.sum() * 0.01, f"${v / 1000:,.1f}k", ha="center", color=INK2)
ax.bar(["Total"], [levers.sum()], color=INK, width=0.6)
ax.text(len(levers), levers.sum() * 1.01, f"${levers.sum() / 1000:,.0f}k / yr", ha="center", color=INK, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
ax.set_ylabel("Extra gross profit per year")
ax.set_title(f"About ${levers.sum() / 1000:,.0f}k/yr of gross profit, mostly from the price list "
             f"(dark = measured from data, light = estimate)")
ax.text(0.01, 0.97, f"Price rises still pay off unless volume falls more than "
        f"{breakeven['Garden']:.0%} (Garden) / {breakeven['Uptown']:.0%} (Uptown).\n"
        f"The 15% promo moved volume by ~0%.", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK2)
ax.grid(axis="x", visible=False)
finish(fig, "pitch_6_value")
