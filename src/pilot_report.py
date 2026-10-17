"""
Summarise a Stories prep-planner pilot log (exported from app/planner.html or filled by hand
using data/pilot/pilot_log_template.csv).

    python src/pilot_report.py data/pilot/my_pilot_log.csv
    python src/pilot_report.py data/pilot/my_pilot_log.csv --json reports/pilot_summary.json

What it reports, per arm (planner / current / shadow):
    item-days, units prepared, sold, left at close (waste), waste %, sell-out rate and median
    sell-out time, overrides, and gross contribution per item-day:
        sold x price - prepared x cost - batches x setup cost - top-up units x extra cost

Planner vs current: difference in mean contribution per item-day with a 95% bootstrap interval
that resamples whole branch-days (items on the same day are not independent). With fewer than
20 branch-days per arm it prints the numbers but says the comparison is not decision-grade.

Unmet demand is not observed in a pilot (customers who found the shelf empty are not recorded),
so service is reported as sell-out rate and time, never as a fill rate. Gross contribution is
not net profit.

Requires: pandas, numpy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["date", "branch", "product", "arm", "prepared_opening", "sold", "unsold_end"]
MIN_DAYS = 20


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing required columns {missing}")
    for c in ["recommended_opening", "prepared_opening", "remaining_at_16", "recommended_topup", "prepared_topup",
              "sold", "unsold_end", "unit_price", "unit_cost", "setup_usd_per_batch", "extra_usd_per_topup_unit"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["prepared_topup"] = df["prepared_topup"].fillna(0)
    df["prepared"] = df["prepared_opening"] + df["prepared_topup"]
    bad = df["sold"] + df["unsold_end"] > df["prepared"]
    if bad.any():
        print(f"warning: {int(bad.sum())} rows have sold + left at close > prepared; check them")
    df["batches"] = 1 + (df["prepared_topup"] > 0)
    df["contribution"] = (df["sold"] * df["unit_price"] - df["prepared"] * df["unit_cost"]
                          - df["batches"] * df["setup_usd_per_batch"].fillna(0)
                          - df["prepared_topup"] * df["extra_usd_per_topup_unit"].fillna(0))
    df["sold_out"] = df.get("sold_out_time").notna() & (df.get("sold_out_time").astype(str).str.strip() != "")
    df["override"] = df.get("override_reason").notna() & (df.get("override_reason").astype(str).str.strip() != "")
    df["branch_day"] = df["branch"].astype(str) + "|" + df["date"].astype(str)
    return df


def summarise(df):
    g = df.groupby("arm")
    out = pd.DataFrame({
        "item_days": g.size(),
        "branch_days": g["branch_day"].nunique(),
        "prepared": g["prepared"].sum(),
        "sold": g["sold"].sum(),
        "waste": g["unsold_end"].sum(),
        "sold_out_rate": g["sold_out"].mean(),
        "overrides": g["override"].sum(),
        "contribution_per_item_day": g["contribution"].mean(),
    })
    out["waste_pct"] = out["waste"] / out["prepared"]
    t = pd.to_datetime(df["sold_out_time"].astype(str).str.slice(0, 5), format="%H:%M", errors="coerce")
    minutes = (t.dt.hour * 60 + t.dt.minute).where(df["sold_out"])
    med = minutes.groupby(df["arm"]).median()
    out["median_sold_out_time"] = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" if pd.notna(m) else ""
                                   for m in med.reindex(out.index)]
    return out


def compare(df, a="planner", b="current", n_boot=5000, seed=490):
    if not {a, b} <= set(df["arm"]):
        return None
    rng = np.random.default_rng(seed)
    days = {arm: df[df["arm"] == arm].groupby("branch_day")["contribution"].agg(["sum", "size"]) for arm in (a, b)}
    point = df.loc[df["arm"] == a, "contribution"].mean() - df.loc[df["arm"] == b, "contribution"].mean()
    boots = []
    for _ in range(n_boot):
        means = []
        for arm in (a, b):
            d = days[arm]
            pick = d.iloc[rng.integers(0, len(d), len(d))]
            means.append(pick["sum"].sum() / pick["size"].sum())
        boots.append(means[0] - means[1])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    n = min(len(days[a]), len(days[b]))
    return {"difference_per_item_day": point, "ci95": [lo, hi], "branch_days_min": n,
            "decision_grade": bool(n >= MIN_DAYS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    df = load(args.log)
    summary = summarise(df)
    pd.set_option("display.width", 200)
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))
    cmp = compare(df)
    if cmp:
        print(f"\nplanner - current: ${cmp['difference_per_item_day']:+.2f} per item-day "
              f"(95% CI ${cmp['ci95'][0]:+.2f} to ${cmp['ci95'][1]:+.2f}, {cmp['branch_days_min']} branch-days in smaller arm)")
        if not cmp["decision_grade"]:
            print(f"Not decision-grade: fewer than {MIN_DAYS} branch-days per arm.")
        print("Compare waste % and sell-out rate too: a profit gain that comes from selling out earlier is a service cost.")
    else:
        print("\nNo planner-vs-current comparison: need both arms in the log.")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary.reset_index().to_dict(orient="records"),
                                         "comparison": cmp}, indent=2, default=float))
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
