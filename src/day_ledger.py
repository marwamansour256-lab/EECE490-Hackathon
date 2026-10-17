"""
Interim daily ledger for the demo's Day explorer tab (teammate 2).

EXTERNAL_CONTEXT.md asks the interface to show, for any date and branch: a traffic/basket/margin
waterfall, weather, menu timing, source/evidence status and unexplained demand. The Codex daily
dossiers (src/day_explanations.py) are not yet published, so this script builds an interim,
clearly labelled version from files that are in the repo. When the Codex dossiers are published,
the app should read them instead; the record schema below is kept close to their description.

For every date x {All branches, Garden, Harbour, Uptown}:
  * actual orders, units, revenue, recorded gross profit
  * expected orders: Poisson model (branch x weekday + branch growth) trained on Jan 1-Mar 31 only,
    frozen; Jan-Mar values are retrospective fits, Apr-Jun are forecasts made at Mar 31
  * reference basket (units/order) and gross profit per unit: last 28 training dates (Mar 4-31)
  * financial bridge: GP - GP_ref split exactly into traffic, basket and margin/mix contributions
    with a symmetric three-factor Shapley decomposition of GP = orders x units/order x GP/unit.
    This is an accounting identity, not causal attribution.
  * unexplained orders: actual - expected, with a Poisson z-score
  * menu timing: share of orders by daypart vs the training reference for that branch/weekday
  * observed weather at the three candidate locations (realised weather, not a pre-opening forecast;
    branch locations are unverified)
  * events active that day from data/external/external_events.json (status and exposure as recorded
    by teammate 1), plus the notebook 03 holiday list and Ramadan

Run from the repo root:
    python src/day_ledger.py
Outputs:
    app/day_data.js                 window.DAY_DATA, read by app/planner.html
    reports/day_ledger.csv          one row per date x branch (no event text)

Requires: pandas, numpy
"""

import itertools
import json
from datetime import date
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "stories_transactions.csv").exists())
EXT = ROOT / "data" / "external"
R, GP = "Line Revenue (USD)", "Line Gross Profit (USD)"
TRAIN_END = pd.Timestamp("2026-03-31")
REF_DAYS = 28
WEATHER_FILES = {"Beirut": "beirut_weather_2026H1.json", "Batroun": "batroun_weather_2026H1.json",
                 "Broummana": "broummana_weather_2026H1.json"}
HOLIDAYS = {"2026-01-01": "New Year's Day", "2026-01-06": "Armenian Christmas", "2026-02-09": "St Maroun Day",
            "2026-02-14": "Rafik Hariri Memorial", "2026-03-20": "Eid al-Fitr", "2026-03-21": "Eid al-Fitr, day 2",
            "2026-03-25": "Annunciation", "2026-04-03": "Good Friday (Western)", "2026-04-05": "Easter (Western)",
            "2026-04-10": "Good Friday (Orthodox)", "2026-04-12": "Easter (Orthodox)", "2026-05-01": "Labour Day",
            "2026-05-25": "Resistance and Liberation Day", "2026-05-27": "Eid al-Adha",
            "2026-05-29": "Eid al-Adha, day 2", "2026-06-16": "Hijri New Year", "2026-06-26": "Ashura"}
RAMADAN = (pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19"))
DAYPARTS = ["morning", "afternoon", "evening"]

# %% Load
df = pd.read_csv(ROOT / "data" / "stories_transactions.csv")  # raw lines retained (AGENTS.md)
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = df["Time"].str[:2].astype(int)
df["daypart"] = pd.cut(df["hour"], [0, 10, 17, 24], include_lowest=True, labels=DAYPARTS).astype(str)
o = df.groupby("Transaction ID").agg(date=("Date", "first"), branch=("Branch", "first"),
                                     daypart=("daypart", "first"), units=("Quantity", "sum"),
                                     rev=(R, "sum"), gp=(GP, "sum"))
DAYS = pd.date_range(df["Date"].min(), df["Date"].max())
BRANCHES = sorted(o["branch"].unique())

# %% Expected orders: frozen Poisson trained on Jan-Mar
idx = pd.MultiIndex.from_product([BRANCHES, DAYS], names=["branch", "date"])
y = o.groupby(["branch", "date"]).size().reindex(idx, fill_value=0).to_numpy(float)
bb, dd = idx.get_level_values("branch"), idx.get_level_values("date")
tt = ((dd - DAYS[0]).days / len(DAYS)).to_numpy()
X = np.column_stack([c for br in BRANCHES for c in
                     [(bb == br).astype(float), (bb == br) * tt] + [(bb == br) * (dd.dayofweek == k) for k in range(1, 7)]])
train = dd <= TRAIN_END
beta, eta = np.zeros(X.shape[1]), np.log(np.maximum(y[train], 0.5))
Xt, yt = X[train], y[train]
for _ in range(60):
    mu = np.exp(eta)
    new = np.linalg.solve((Xt.T * mu) @ Xt + 1e-6 * np.eye(X.shape[1]), (Xt.T * mu) @ (eta + (yt - mu) / mu))
    done = np.abs(new - beta).max() < 1e-9
    beta, eta = new, Xt @ new
    if done:
        break
expected = pd.Series(np.exp(X @ beta), idx)

# %% Reference basket and margin per unit: last 28 training dates
ref_window = (o["date"] > TRAIN_END - pd.Timedelta(days=REF_DAYS)) & (o["date"] <= TRAIN_END)
ref = o[ref_window]


def ref_of(frame):
    return {"units_per_order": float(frame["units"].sum() / len(frame)),
            "gp_per_unit": float(frame["gp"].sum() / frame["units"].sum())}


REF = {b: ref_of(ref[ref["branch"] == b]) for b in BRANCHES} | {"All branches": ref_of(ref)}

# Training-period daypart mix by branch and weekday (menu timing reference)
tr = o[o["date"] <= TRAIN_END].assign(dow=lambda x: x["date"].dt.dayofweek)
mix_ref = {}
for (b, w), g in tr.groupby(["branch", "dow"]):
    mix_ref[(b, w)] = g["daypart"].value_counts(normalize=True).reindex(DAYPARTS, fill_value=0).to_dict()
for w, g in tr.groupby("dow"):
    mix_ref[("All branches", w)] = g["daypart"].value_counts(normalize=True).reindex(DAYPARTS, fill_value=0).to_dict()


def shapley3(x0, x1):
    """Exact Shapley split of prod(x1) - prod(x0) across three multiplicative factors."""
    n = len(x0)
    out = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        cur = list(x0)
        for i in perm:
            before = np.prod(cur)
            cur[i] = x1[i]
            out[i] += np.prod(cur) - before
    return out / factorial(n)


# %% Weather and events
weather = {}
for loc, f in WEATHER_FILES.items():
    w = pd.DataFrame(json.loads((EXT / f).read_text())["daily"])
    weather[loc] = w.set_index(pd.to_datetime(w.pop("time")))

ev_raw = json.loads((EXT / "external_events.json").read_text(encoding="utf-8"))
events = ev_raw if isinstance(ev_raw, list) else list(ev_raw.values())[0] if len(ev_raw) == 1 else ev_raw.get("events", [])
if isinstance(events, dict):
    events = list(events.values())


def day_of(s):
    return pd.Timestamp(str(s)[:10]) if s else None


def events_on(d):
    out = []
    for e in events:
        a, b = day_of(e.get("event_start")), day_of(e.get("event_end")) or day_of(e.get("event_start"))
        if a is not None and a <= d <= b:
            if e.get("event_id") == "HOLIDAYS-2026-H1-NATIONAL":
                continue  # covered day by day below
            out.append({"id": e.get("event_id"), "status": e.get("validation_status"),
                        "exposure": e.get("branch_exposure"), "mechanism": e.get("mechanism"),
                        "known_before": e.get("known_before_event"), "source": e.get("source_url"),
                        "association": e.get("association_in_data")})
    key = d.strftime("%Y-%m-%d")
    if key in HOLIDAYS:
        out.append({"id": f"HOLIDAY: {HOLIDAYS[key]}", "status": "calendar (notebook 03 list)",
                    "exposure": "national calendar; branch hours unknown", "mechanism": "public holiday",
                    "known_before": True, "source": None, "association": None})
    if RAMADAN[0] <= d <= RAMADAN[1]:
        out.append({"id": "RAMADAN 2026", "status": "calendar", "exposure": "national calendar",
                    "mechanism": "fasting month", "known_before": True, "source": None, "association": None})
    return out


# %% Build records
records = []
csv_rows = []
for d in DAYS:
    od = o[o["date"] == d]
    dow = d.dayofweek
    period = "train (retrospective fit)" if d <= TRAIN_END else "test (forecast frozen at Mar 31)"
    ev = events_on(d)
    wx = {loc: {"tmax": round(float(w.loc[d, "temperature_2m_max"]), 1),
                "rain_mm": round(float(w.loc[d, "precipitation_sum"]), 1),
                "wind_max": round(float(w.loc[d, "wind_speed_10m_max"]), 1)} for loc, w in weather.items()}
    for b in ["All branches"] + BRANCHES:
        ob = od if b == "All branches" else od[od["branch"] == b]
        exp = float(expected.xs(d, level="date").sum() if b == "All branches" else expected[(b, d)])
        n, units, gp, rev = len(ob), float(ob["units"].sum()), float(ob["gp"].sum()), float(ob["rev"].sum())
        r = REF[b]
        x0 = [exp, r["units_per_order"], r["gp_per_unit"]]
        x1 = [n, units / n if n else r["units_per_order"], gp / units if units else r["gp_per_unit"]]
        bridge = shapley3(x0, x1)
        mix = ob["daypart"].value_counts(normalize=True).reindex(DAYPARTS, fill_value=0).to_dict()
        rec = {
            "date": d.strftime("%Y-%m-%d"), "weekday": d.day_name(), "branch": b, "period": period,
            "orders": n, "expected_orders": round(exp, 1), "units": units, "revenue": round(rev, 2),
            "gross_profit": round(gp, 2), "reference_gross_profit": round(float(np.prod(x0)), 2),
            "bridge": {"traffic": round(float(bridge[0]), 2), "basket": round(float(bridge[1]), 2),
                       "margin_mix": round(float(bridge[2]), 2)},
            "unexplained_orders": round(n - exp, 1), "z": round((n - exp) / np.sqrt(exp), 2) if exp > 0 else None,
            "daypart_share": {k: round(float(v), 3) for k, v in mix.items()},
            "daypart_share_ref": {k: round(float(v), 3) for k, v in mix_ref[(b, dow)].items()},
        }
        records.append(rec)
        csv_rows.append({k: v for k, v in rec.items() if k not in ("bridge", "daypart_share", "daypart_share_ref")}
                        | {f"bridge_{k}": v for k, v in rec["bridge"].items()})
    records[-4]["weather"] = wx  # attach once per date (on the "All branches" record)
    records[-4]["events"] = ev

meta = {
    "generated": str(date.today()), "generator": "src/day_ledger.py (interim, teammate 2)",
    "status": "Interim app-computed ledger. Replace with the Codex daily dossiers when published.",
    "expected_orders_model": "Poisson branch x weekday + branch growth, trained Jan 1-Mar 31, frozen",
    "reference": f"basket and GP/unit from the last {REF_DAYS} training dates (Mar 4-31), per branch",
    "bridge": "symmetric 3-factor Shapley of GP = orders x units/order x GP/unit vs reference; accounting, not causal",
    "weather": "Open-Meteo realised daily weather at three candidate locations; branch locations unverified; not a pre-opening forecast",
    "events": "data/external/external_events.json (teammate 1) + notebook 03 holidays + Ramadan; exposure as recorded",
    "references": REF,
}
(ROOT / "app" / "day_data.js").write_text(
    "// Generated by src/day_ledger.py. Do not edit by hand; rerun the script.\n"
    f"window.DAY_DATA = {json.dumps({'meta': meta, 'records': records}, separators=(',', ':'))};\n", encoding="utf-8")
pd.DataFrame(csv_rows).to_csv(ROOT / "reports" / "day_ledger.csv", index=False)
print(f"saved app/day_data.js ({len(records)} records) and reports/day_ledger.csv")
for d, b in [("2026-06-24", "All branches"), ("2026-05-01", "Harbour"), ("2026-04-08", "All branches")]:
    r = next(x for x in records if x["date"] == d and x["branch"] == b)
    print(d, b, f"GP {r['gross_profit']:.0f} vs ref {r['reference_gross_profit']:.0f}: bridge {r['bridge']}, "
                f"orders {r['orders']} vs {r['expected_orders']} (z {r['z']})")
