"""Connect basket offers, frozen demand, preparation and day-level evidence.

This orchestration loads committed models, never trains on a requested date's
sales. It does not assume that displaying an offer increases demand. Incremental
acceptance is an explicit scenario, zero by default, and is kept out of forecasts.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

from basket_recommender import recommend
from demand_model import PERISHABLES
from profit_policy import optimize
from strict_model import category_mean, share_column

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def prep_artifact():
    return joblib.load(ROOT/"models/strict_half_model.joblib")


def prep_card(date, branch, product, setup=.25, premium=.10, fill_floor=.90, artifact=None):
    artifact = artifact if artifact is not None else prep_artifact()
    stamp = pd.Timestamp(date)
    if pd.isna(stamp) or stamp <= pd.Timestamp(artifact["train_end"]):
        raise ValueError("Choose a valid scenario date after the March 31 training cutoff")
    params = artifact["sku_parameters"]
    selected = params[params.Branch.eq(branch) & params.Product.eq(product)]
    if len(selected) != 1:
        raise ValueError("Unknown branch/product in preparation model")
    row = selected.iloc[0]
    frame = pd.DataFrame([dict(Date=stamp, Branch=branch, Category=row.Category)])
    method = artifact["selected_method"]
    mean = float(category_mean(artifact, frame, method)[0]) * float(row[share_column(method)])
    card = dict(date=str(stamp.date()),branch=branch,product=product,category=row.Category,
        status="Historical scenario" if stamp <= pd.Timestamp("2026-06-30") else "Extrapolation beyond observed window; not a validated operational forecast",
        trained_through=artifact["train_end"],method=method,forecast_mean_units=mean,
        recommendation_uplift_added_to_forecast=False,test_sales_used_in_forecast=False,
        preparation_applicable=row.Category in PERISHABLES,
        assumptions=dict(setup_usd_per_sku=setup,extra_usd_per_topup_unit=premium,expected_fill_floor=fill_floor,
                         instant_replenishment=True,capacity_unconstrained=True))
    if row.Category not in PERISHABLES:
        card["preparation_note"] = "Made-to-order/nonperishable category: demand reference only; no pastry batch instruction"
        return card
    policy = optimize(mean,float(row.early_share),float(row.price),float(row.cost),setup=setup,premium=premium,fill_floor=fill_floor)
    opening = policy["opening_units"]
    card.update(opening_prepare_units=opening,one_batch_comparator_units=policy["one_batch_units"],
        at16_topup_by_remaining_stock={str(i):int(q) for i,q in enumerate(policy["replenish_by_remaining_stock"]) if i<=opening},
        expected_two_stage_contribution=policy["expected_two_stage_contribution"],
        expected_one_batch_contribution=policy["expected_one_batch_contribution"])
    return card


def profit_scenario(eligible_orders, incremental_acceptance, unit_margin, discount_per_extra=0,
                    waste=0, labor=0, implementation=0):
    values=[eligible_orders,incremental_acceptance,unit_margin,discount_per_extra,waste,labor,implementation]
    import math
    if any(not math.isfinite(v) or v<0 for v in values) or not 0<=incremental_acceptance<=1:
        raise ValueError("Scenario inputs must be finite/nonnegative and incremental acceptance between 0 and 1")
    extra=eligible_orders*incremental_acceptance
    gross=extra*(unit_margin-discount_per_extra)
    denominator=eligible_orders*(unit_margin-discount_per_extra)
    return dict(status="Editable assumption scenario; not model-estimated incremental acceptance",
        eligible_orders=eligible_orders,incremental_acceptance_difference=incremental_acceptance,
        additional_add_ons=extra,unit_margin_usd=unit_margin,discount_per_extra_add_on_usd=discount_per_extra,
        incremental_waste_usd=waste,extra_labor_usd=labor,implementation_usd=implementation,
        contribution_before_extra_costs_usd=gross,incremental_contribution_scenario_usd=gross-waste-labor-implementation,
        break_even_incremental_acceptance=(waste+labor+implementation)/denominator if denominator>0 else None,
        caveat="Assumes no cannibalization/substitution and enough stock; offer accuracy is not incremental acceptance")


def plan(date,branch,basket,part="all",cohort="all",unavailable=(),mode="complementary",
         eligible_orders=100,incremental_acceptance=0,waste=0,labor=0,implementation=0,discount=0):
    payload=json.loads((ROOT/"models/basket_recommender.json").read_text(encoding="utf-8"))
    offers=recommend(payload,branch,basket,part,cohort,unavailable,mode=mode)
    cards=[prep_card(date,branch,r["product"]) for r in offers["recommendations"]]
    first=offers["recommendations"][0] if offers["recommendations"] else None
    scenario=profit_scenario(eligible_orders,incremental_acceptance,first["unit_margin_usd"] if first else 0,
        discount,waste,labor,implementation)
    # Retrospective evidence is deliberately nested outside the decision inputs.
    ledger_path=ROOT/"reports/daily_why_ledger.json"
    explanation=None
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
        explanation=next((r for r in ledger["records"] if r["date"]==date and r["branch"]==branch),None)
    return dict(schema_version=1,status="Integrated historical decision demo; business uplift requires a controlled pilot",
        date=date,branch=branch,offers=offers,preparation=cards,
        profit_scenario_for_top_offer=scenario,
        retrospective_diagnostic_not_used_to_select_offer_or_prepare=explanation,
        model_boundaries=["Offers and preparation are frozen Jan-Mar artifacts",
            "Remaining stock determines top-up via each preparation card's lookup",
            "No assumed upsell is silently added to demand or contribution forecasts",
            "Observed external-event diagnostics do not become advance forecast inputs"])


def export_preparation(date="2026-06-30"):
    artifact=prep_artifact()
    cards={}
    for row in artifact["sku_parameters"].itertuples(index=False):
        if row.Category in PERISHABLES:
            cards[f"{row.Branch}|{row.Product}"]=prep_card(date,row.Branch,row.Product,artifact=artifact)
    payload=dict(schema_version=1,status="Historical preparation scenarios, not live stock or current-day forecasts",
        date=date,cards=cards,contract="key=branch|product; opening_prepare_units and at16_topup_by_remaining_stock as existing prep contract")
    encoded=json.dumps(payload,separators=(",", ":"),allow_nan=False)
    (ROOT/"app/preparation_data.js").write_text("window.STORIES_PREPARATION="+encoded+";\n",encoding="utf-8")
    (ROOT/"reports/growth_preparation_cards.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(date=date,cards=len(cards),output="app/preparation_data.js")))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date",default="2026-06-30")
    p.add_argument("--branch",default="Harbour")
    p.add_argument("--item",action="append",default=[])
    p.add_argument("--unavailable",action="append",default=[])
    p.add_argument("--daypart",default="all",choices=["all","morning","midday","evening"])
    p.add_argument("--cohort",default="all",choices=["all","regular"])
    p.add_argument("--mode",default="complementary",choices=["unrestricted","complementary"])
    p.add_argument("--eligible-orders",type=int,default=100)
    p.add_argument("--incremental-acceptance",type=float,default=0)
    p.add_argument("--discount",type=float,default=0)
    p.add_argument("--waste",type=float,default=0)
    p.add_argument("--labor",type=float,default=0)
    p.add_argument("--implementation",type=float,default=0)
    p.add_argument("--export-preparation",action="store_true")
    p.add_argument("--output",help="Optional JSON output file")
    a=p.parse_args()
    try:
        if a.export_preparation:
            export_preparation(a.date)
            return
        result=plan(a.date,a.branch,a.item,a.daypart,a.cohort,a.unavailable,a.mode,
            a.eligible_orders,a.incremental_acceptance,a.waste,a.labor,a.implementation,a.discount)
    except ValueError as exc:
        p.error(str(exc))
    encoded=json.dumps(result,indent=2,allow_nan=False)
    if a.output:
        Path(a.output).write_text(encoded,encoding="utf-8")
    else:
        print(encoded)


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
