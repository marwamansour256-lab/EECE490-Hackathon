"""
Metric reconciliation (ANALYSIS.md team action: "Omar, with Marwa: agree one definition for each
number that differs between documents").

For every headline number that appears with different values across the README, ANALYSIS.md,
notebooks 01-08 and the pitch scripts, this script recomputes each variant exactly as it was
defined where it appears, plus the proposed single definition, on both raw lines (AGENTS.md rule:
retain repeated lines) and with the 132 repeated lines dropped (notebook 01's rule).

Run from the repo root:
    python src/metric_reconciliation.py
Outputs:
    reports/metric_reconciliation.json     every variant, value and source
    reports/metric_reconciliation.md        the human-readable table (regenerated each run)

Requires: pandas, numpy
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
R, GP = "Line Revenue (USD)", "Line Gross Profit (USD)"
PROMO_STARTS = [pd.Timestamp("2026-03-12"), pd.Timestamp("2026-05-12")]  # 8 days each, Harbour only
PROMO_CATS = ["Cold Coffee", "Desserts"]

raw = pd.read_csv(ROOT / "data" / "stories_transactions.csv")
raw["Date"] = pd.to_datetime(raw["Date"])
DATASETS = {"raw lines (AGENTS.md rule)": raw, "132 repeated lines dropped (notebook 01 rule)": raw.drop_duplicates()}


def orders_of(df):
    return df.groupby("Transaction ID").agg(
        date=("Date", "first"), branch=("Branch", "first"), customer=("Customer ID", "first"),
        lines=("Product", "size"), products=("Product", "nunique"), units=("Quantity", "sum"),
        rev=(R, "sum"), gp=(GP, "sum"), disc10=("Discount Rate", lambda s: bool((s == 0.10).any())))


def promo_window(d):
    return any(s <= d <= s + pd.Timedelta(days=7) for s in PROMO_STARTS)


def metrics(df):
    o = orders_of(df)
    o["dow"] = o["date"].dt.dayofweek
    m = {}

    # 1. Single-item orders
    m["single_item"] = {
        "one product line (notebook 01: '54% of orders contain a single line')": float((o["lines"] == 1).mean()),
        "one unit in total (pitch_visuals.py: '51% of orders are a single item')": float((o["units"] == 1).mean()),
        "one distinct product, any quantity (notebook 06: '47% contain two or more different products')":
            float((o["products"] == 1).mean()),
    }

    # 2. Retention / repeat
    k = o[o["customer"].notna()].copy()
    k["month"] = k["date"].dt.month
    active = {mm: set(g["customer"]) for mm, g in k.groupby("month")}
    early = active[1] | active[2]
    late = active[5] | active[6]
    first = k.groupby("customer")["date"].min().dt.month
    jan = set(first[first == 1].index)
    month_repeat = [len(active[mm] & active[mm + 1]) / len(active[mm]) for mm in range(1, 6)]
    m["retention"] = {
        "Jan-Feb customers seen again in May-Jun (pitch / README: 91%)": len(early & late) / len(early),
        "January first-seen cohort active in June (notebook 04 cohort heatmap, Jan row, Jun column)":
            len(jan & active[6]) / len(jan),
        "month-to-month repeat: active in month m who are active again in m+1 (mean of Jan-Feb ... May-Jun)":
            float(np.mean(month_repeat)),
        "month-to-month repeat by month": [round(x, 4) for x in month_repeat],
    }

    # 3. Harbour 15% promotion effect (Cold Coffee + Desserts)
    df = df.assign(promo=df["Date"].map(promo_window), month=df["Date"].dt.month)
    pc = df[df["Category"].isin(PROMO_CATS)]
    in_months = pc["month"].isin([3, 5])

    def per_day(frame, col):
        return frame.groupby("Branch").apply(lambda g: g[col].sum() / g["Date"].nunique(), include_groups=False)

    def windowed(frame):
        rows = []
        for s in PROMO_STARTS:
            for lab, a, b in [("before", s - pd.Timedelta(days=14), s - pd.Timedelta(days=1)),
                              ("promo", s, s + pd.Timedelta(days=7)),
                              ("after", s + pd.Timedelta(days=8), s + pd.Timedelta(days=21))]:
                rows.append(frame[(frame["Date"] >= a) & (frame["Date"] <= b)].assign(w=lab))
        return pd.concat(rows)

    w = windowed(pc)
    q = {lab: per_day(w[w["w"] == lab], "Quantity") for lab in ["before", "promo", "after"]}
    g = {lab: per_day(w[w["w"] == lab], GP) for lab in ["before", "promo", "after"]}
    same_q = {"promo": per_day(pc[in_months & pc["promo"]], "Quantity"), "other": per_day(pc[in_months & ~pc["promo"]], "Quantity")}
    same_g = {"promo": per_day(pc[in_months & pc["promo"]], GP), "other": per_day(pc[in_months & ~pc["promo"]], GP)}

    def did(a, b):
        """Harbour change minus the average change at Garden and Uptown (relative)."""
        ch = a / b - 1
        return float(ch["Harbour"] - ch[["Garden", "Uptown"]].mean())

    store = df[df["Branch"] == "Harbour"]
    store_m = store[store["month"].isin([3, 5])]
    p15 = df[df["Discount Rate"] == 0.15]
    m["promo"] = {
        "promo-category units/day, Harbour, promo vs other March+May days (notebook 07: '+1%')":
            float(same_q["promo"]["Harbour"] / same_q["other"]["Harbour"] - 1),
        "promo-category units/day, Harbour, promo vs 2 weeks before (pitch: 33.9 -> 33.8)":
            float(q["promo"]["Harbour"] / q["before"]["Harbour"] - 1),
        "promo-category gross profit/day, Harbour, promo vs 2 weeks before (pitch: '$130 -> $100, -23%')":
            float(g["promo"]["Harbour"] / g["before"]["Harbour"] - 1),
        "whole-store gross profit/day, Harbour, promo vs other March+May days (notebook 01: 'about -4%')":
            float((store_m[store_m["promo"]][GP].sum() / store_m[store_m["promo"]]["Date"].nunique())
                  / (store_m[~store_m["promo"]][GP].sum() / store_m[~store_m["promo"]]["Date"].nunique()) - 1),
        "PROPOSED units: difference-in-differences vs Garden+Uptown, promo vs other March+May days":
            did(same_q["promo"], same_q["other"]),
        "PROPOSED gross profit: difference-in-differences vs Garden+Uptown, promo vs other March+May days":
            did(same_g["promo"], same_g["other"]),
        "check, 2-week windows: units DiD": did(q["promo"], q["before"]),
        "check, 2-week windows: gross profit DiD": did(g["promo"], g["before"]),
        "discount given away, USD (notebook 07 / pitch: about $470)":
            float((p15["Quantity"] * p15["Unit Price (USD)"] * 0.15).sum()),
    }

    # 4. Tuesday
    n_days = o.groupby("dow")["date"].nunique()
    per = o.groupby("dow").size() / n_days
    tot = o.groupby("dow").size()
    m["tuesday"] = {
        "per-day orders, Tuesday / mean of the other six days incl. the weekend (ANALYSIS.md: 1.66)":
            float(per[1] / per.drop(1).mean()),
        "ratio of weekday TOTALS, Tuesday / mean of Mon,Wed-Fri totals (closest to notebooks 01, 03: '+55%')":
            float(tot[1] / tot[[0, 2, 3, 4]].mean()),
        "per-day orders, Tuesday / mean of Mon,Wed-Fri (why_drivers.py: 1.53)":
            float(per[1] / per[[0, 2, 3, 4]].mean()),
        "days of each weekday in the data": {int(k_): int(v) for k_, v in n_days.items()},
    }

    # 5. The standing 10% discount
    m["discount10"] = {
        "share of LINES with 10% (notebook 07, ANALYSIS.md: '5.6% of lines')": float((df["Discount Rate"] == 0.10).mean()),
        "share of ORDERS with any 10% line (notebook 01 text says 'about 6% of orders'; its own table shows 0.08)":
            float(o["disc10"].mean()),
        "10% + 15% discount given away per year, USD (notebook 07: about $4,400)": float(
            2 * (df["Quantity"] * df["Unit Price (USD)"] * df["Discount Rate"]).sum()),
    }

    # 6. Order growth Jan -> Jun
    mo = o.groupby(o["date"].dt.month).agg(n=("rev", "size"), days=("date", "nunique"))
    m["growth"] = {
        "monthly order totals, Jan -> Jun (notebooks: 'about +50%')": float(mo.loc[6, "n"] / mo.loc[1, "n"] - 1),
        "orders per day, Jan -> Jun": float((mo.loc[6, "n"] / mo.loc[6, "days"]) / (mo.loc[1, "n"] / mo.loc[1, "days"]) - 1),
    }
    return m


results = {name: metrics(d) for name, d in DATASETS.items()}

M0 = results[next(iter(results))]
SI = M0["single_item"]
T = M0["tuesday"]
D10 = M0["discount10"]
GR = M0["growth"]
PROPOSALS = [
    ("single_item", "Single-item orders",
     "one distinct product, any quantity (notebook 06: '47% contain two or more different products')",
     "Share of orders that contain only one distinct product (any quantity). This is the audience for a "
     "different-product add-on. Report as '{v:.0%} of orders contain only one product'. The one-line "
     f"({list(SI.values())[0]:.1%}) and one-unit ({list(SI.values())[1]:.1%}) figures answer different questions; "
     "do not use them as the headline."),
    ("retention", "Retention",
     "month-to-month repeat: active in month m who are active again in m+1 (mean of Jan-Feb ... May-Jun)",
     "Month-to-month repeat rate of identified customers: of those who ordered in a month, the share who order "
     "again the next month ({v:.0%} on average). Call the 91% figure 'overlap between Jan-Feb and May-Jun "
     "customers', never 'retention'. 'New' customers are first OBSERVED, not first ever (README)."),
    ("promo", "Promotion profit effect",
     "PROPOSED gross profit: difference-in-differences vs Garden+Uptown, promo vs other March+May days",
     "Harbour promo-category gross profit per day during the 16 promo days vs the other days of March and May, "
     "minus the same change at Garden and Uptown: {v:+.0%}. Report it together with the units version and the "
     "$471 discount given away, and say it is 16 days, two categories and one branch, so it is not an elasticity."),
    ("tuesday", "Tuesday effect",
     "per-day orders, Tuesday / mean of Mon,Wed-Fri (why_drivers.py: 1.53)",
     "Orders per Tuesday divided by the average orders per day on Monday and Wednesday-Friday: {v:.2f}x "
     "('Tuesday is about {p:.0%} busier than a normal weekday'). Comparing with weekdays keeps the weekend dip "
     "out of the Tuesday effect. ANALYSIS.md's "
     f"{list(T.values())[0]:.2f}x is also correct but compares with all six other days, including the quiet "
     "weekend; if it is used, say 'than the average of the other six days'. Use per-day rates: the period has "
     "25 Wednesdays and 26 of every other weekday."),
    ("discount10", "Standing 10% discount",
     "share of LINES with 10% (notebook 07, ANALYSIS.md: '5.6% of lines')",
     "Always say lines: '{v:.1%} of product lines get 10% off'. The order-level figure "
     f"({list(D10.values())[1]:.1%} of orders contain a 10% line) is a different unit; give it only with the "
     "word 'orders'. Notebook 01's 'about 6% of orders' does not match its own table (0.08)."),
    ("growth", "Order growth",
     "orders per day, Jan -> Jun",
     "Orders per day, January vs June: {v:+.0%}. Monthly totals "
     f"({list(GR.values())[0]:+.0%}) understate it slightly because January has 31 days and June 30."),
]

main, alt = list(DATASETS)
lines_md = [
    "# Metric reconciliation",
    "",
    f"*Generated by `src/metric_reconciliation.py` on {date.today()}. Proposed by Marwa (Claude Code teammate 2) "
    "for the ANALYSIS.md team action \"agree one definition for each number that differs between documents\". "
    "**Status: proposed, pending agreement with Omar (teammate 1).** Values use raw lines (AGENTS.md); the "
    "deduplicated value is shown where it differs.*",
    "",
    "| Metric | Proposed definition and wording | Value (raw lines) | Value (repeats dropped) |",
    "|---|---|---:|---:|",
]
summary = {}
for key, title, variant, wording in PROPOSALS:
    v, v2 = results[main][key][variant], results[alt][key][variant]
    fmt = (lambda x: f"{x:.2f}x") if key == "tuesday" else (lambda x: f"${x:,.0f}") if "USD" in variant else (lambda x: f"{x:+.1%}" if key in ("promo", "growth") else f"{x:.1%}")
    text = wording.format(v=v, p=v - 1)
    lines_md.append(f"| **{title}** | {text} | {fmt(v)} | {fmt(v2)} |")
    summary[key] = {"title": title, "proposed_variant": variant, "value_raw": v, "value_dedup": v2, "wording": text}

lines_md += ["", "## Every variant found in the documents", ""]
for key, title, *_ in PROPOSALS:
    lines_md += [f"### {title}", "", "| Definition (where it appears) | Raw lines | Repeats dropped |", "|---|---:|---:|"]
    for var, val in results[main][key].items():
        if isinstance(val, (list, dict)):
            continue
        v2 = results[alt][key][var]
        if "USD" in var:
            f = lambda x: f"${x:,.0f}"
        elif key == "tuesday":
            f = lambda x: f"{x:.3f}x"
        elif key in ("promo", "growth"):
            f = lambda x: f"{x:+.1%}"
        else:
            f = lambda x: f"{x:.1%}"
        lines_md.append(f"| {var} | {f(val)} | {f(v2)} |")
    extra = {k_: v_ for k_, v_ in results[main][key].items() if isinstance(v_, (list, dict))}
    for k_, v_ in extra.items():
        lines_md.append(f"\n{k_}: `{v_}`")
    lines_md.append("")

lines_md += [
    "## Edits needed once agreed",
    "",
    "- **ANALYSIS.md (Omar):** item 7 says 'Tuesdays at about 1.66 times other days'. That is correct for 'all six "
    "other days'; either keep it and add that wording, or switch to the proposed weekday comparison (1.53x). "
    "Item 6 says 'about half of orders contain a single product'; that matches the proposed definition, so add the exact value.",
    "- **Promotion:** no document should quote the pitch's -23% promo profit figure alone. It compares with the "
    "2 weeks before and has no control branches. The proposed difference-in-differences is the headline, with "
    "units, gross profit and the discount cost reported together.",
    "- **Notebooks 01-08:** stay as the team's original work (AGENTS.md); the corrections live here and in the README.",
    "- **README (shared):** the figures in the 'Why the numbers move' section already use per-day rates; "
    "the old pitch figures (51% single-item, 91% 'retained', -23% promo profit) are superseded by this table.",
    "",
]
(ROOT / "reports" / "metric_reconciliation.md").write_text("\n".join(lines_md), encoding="utf-8")
(ROOT / "reports" / "metric_reconciliation.json").write_text(json.dumps(
    {"run_date": str(date.today()), "status": "proposed by teammate 2, pending teammate 1 agreement",
     "proposals": summary, "all_variants": results}, indent=2, default=float), encoding="utf-8")

for key, s in summary.items():
    print(f"{s['title']:<26} raw {s['value_raw']:.4f}  dedup {s['value_dedup']:.4f}")
print("saved reports/metric_reconciliation.md and .json")
