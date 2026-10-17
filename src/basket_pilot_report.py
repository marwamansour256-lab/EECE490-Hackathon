"""Measure recorded basket-pilot contribution; never invent acceptance or uplift.

One row per assigned eligible order, including controls and declined offers.
Net revenue already contains discounts; do not subtract discounts a second time.
Inference below is conditional whole-branch-day resampling, not verification that
assignment was randomized or proof that an underpowered pilot establishes ROI.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REQUIRED=["experiment_id","experiment_type","date","branch","arm","order_id","eligible","offer_shown",
    "accepted","order_revenue_usd","order_cost_usd","discount_usd","waste_cost_allocated_usd",
    "extra_labor_allocated_usd","other_cost_allocated_usd","stock_available","assignment_probability"]
COSTS=["order_cost_usd","waste_cost_allocated_usd","extra_labor_allocated_usd","other_cost_allocated_usd"]
ARMS={"recommendation":{"current","popular","model"},"layout":{"current","adjacent","separated"}}


def validate(frame):
    missing=set(REQUIRED)-set(frame.columns)
    if missing:
        raise ValueError("Missing pilot columns: "+", ".join(sorted(missing)))
    frame=frame.copy()
    if frame.empty:
        return frame
    if frame[["experiment_id","experiment_type","date","branch","arm","order_id"]].isna().any().any():
        raise ValueError("Experiment, assignment and order identifiers must be recorded")
    if frame.duplicated(["experiment_id","order_id"]).any():
        raise ValueError("Duplicate order in the same experiment")
    frame["date"]=pd.to_datetime(frame.date,errors="raise").dt.strftime("%Y-%m-%d")
    for column in ["eligible","offer_shown","accepted","stock_available"]:
        frame[column]=pd.to_numeric(frame[column],errors="raise")
        if not frame[column].isin([0,1]).all():
            raise ValueError(f"{column} must be explicitly 0 or 1")
    for column in ["order_revenue_usd","discount_usd",*COSTS]:
        frame[column]=pd.to_numeric(frame[column],errors="raise")
        if not np.isfinite(frame[column]).all() or frame[column].lt(0).any():
            raise ValueError(f"{column} must be measured, finite and nonnegative")
    frame["assignment_probability"]=pd.to_numeric(frame.assignment_probability,errors="coerce")
    if frame.assignment_probability.isna().any() or not frame.assignment_probability.between(0,1,inclusive="neither").all():
        raise ValueError("Record randomized arm-assignment probability strictly between 0 and 1")
    if frame.groupby(["experiment_id","branch","date"]).arm.nunique().gt(1).any():
        raise ValueError("An assigned branch-day contains multiple arms")
    if frame.groupby(["experiment_id","branch","date"]).assignment_probability.nunique().gt(1).any():
        raise ValueError("An assigned branch-day contains inconsistent assignment probabilities")
    if "service_seconds" in frame:
        frame["service_seconds"]=pd.to_numeric(frame.service_seconds,errors="raise")
        measured=frame.service_seconds.dropna()
        if not np.isfinite(measured).all() or measured.lt(0).any():
            raise ValueError("Recorded service seconds must be finite/nonnegative; unknown stays empty")
    for experiment,group in frame.groupby("experiment_id"):
        types=group.experiment_type.unique()
        if len(types)!=1 or types[0] not in ARMS or not set(group.arm).issubset(ARMS[types[0]]):
            raise ValueError(f"Invalid or mixed experiment/arms in {experiment}")
    if (frame.accepted.gt(frame.offer_shown) | frame.offer_shown.gt(frame.eligible)).any():
        raise ValueError("Accepted offers require a shown offer and an eligible order")
    if (frame.accepted.gt(frame.stock_available)).any():
        raise ValueError("A paid accepted add-on cannot be recorded as unavailable")
    frame["contribution_usd"]=frame.order_revenue_usd-frame[COSTS].sum(axis=1)
    return frame


def summarize(frame,control="current",draws=2000):
    frame=validate(frame)
    if frame.empty:
        return dict(status="Template only; no pilot outcomes observed",experiments=[],incremental_profit_measured=False)
    reports=[]
    rng=np.random.default_rng(490)
    for name,all_rows in frame.groupby("experiment_id"):
        selected=all_rows[all_rows.eligible.eq(1)].copy()
        if selected.empty:
            reports.append(dict(experiment_id=name,status="No eligible orders"))
            continue
        groups=selected.groupby(["arm","branch","date"]).agg(
            contribution_usd=("contribution_usd","sum"),orders=("order_id","size"),
            offers=("offer_shown","sum"),accepted=("accepted","sum"),unavailable=("stock_available",lambda x:int(x.eq(0).sum())))
        arms={}
        for arm,g in groups.groupby(level="arm"):
            units=len(g)
            offers=int(g.offers.sum())
            arm_orders=selected[selected.arm.eq(arm)]
            service=arm_orders.service_seconds.dropna() if "service_seconds" in arm_orders else pd.Series(dtype=float)
            arms[arm]=dict(eligible_orders=int(g.orders.sum()),branch_days=units,
                contribution_usd=float(g.contribution_usd.sum()),
                contribution_per_eligible_order_usd=float(g.contribution_usd.sum()/g.orders.sum()),
                contribution_per_branch_day_usd=float(g.contribution_usd.mean()),offers_shown=offers,
                accepted=int(g.accepted.sum()),observed_acceptance_per_shown=float(g.accepted.sum()/offers) if offers else None,
                unavailable_eligible_orders=int(g.unavailable.sum()),
                recorded_discount_usd=float(arm_orders.discount_usd.sum()),
                service_seconds_mean=float(service.mean()) if len(service) else None,
                service_measurement_orders=len(service),
                sample_flag="Fewer than20 branch-days; exploratory only" if units<20 else "20+ branch-days is not a guarantee of power or unbiased assignment")
        comparisons=[]
        if control in arms:
            for treatment in sorted(set(arms)-{control}):
                contrast=arms[treatment]["contribution_per_eligible_order_usd"]-arms[control]["contribution_per_eligible_order_usd"]
                sampled={}
                for arm in [control,treatment]:
                    arm_days=groups.xs(arm,level="arm")
                    total_profit=np.zeros(draws)
                    total_orders=np.zeros(draws)
                    for _,branch_days in arm_days.groupby(level="branch"):
                        idx=rng.integers(0,len(branch_days),size=(draws,len(branch_days)))
                        total_profit+=branch_days.contribution_usd.to_numpy()[idx].sum(axis=1)
                        total_orders+=branch_days.orders.to_numpy()[idx].sum(axis=1)
                    sampled[arm]=total_profit/total_orders
                interval=np.quantile(sampled[treatment]-sampled[control],[.025,.975]).tolist()
                comparisons.append(dict(treatment=treatment,control=control,
                    observed_contribution_difference_per_eligible_order_usd=float(contrast),
                    conditional_whole_branch_day_bootstrap_95pct=interval,
                    meaning="Observed arm contrast; causal interpretation requires verified assignment, complete logging and suitable design"))
        reports.append(dict(experiment_id=name,experiment_type=selected.experiment_type.iloc[0],
            excluded_ineligible_rows=int(all_rows.eligible.eq(0).sum()),arms=arms,comparisons=comparisons,
            control_present=control in arms,
            allocation_probability_warning=bool(selected.groupby("arm").assignment_probability.nunique().gt(1).any()),
            design_limits=["Assignment probability is a recorded declaration, not proof of randomization",
                "Keep all assigned eligible orders, including no offer, declined offers and unavailable stock",
                "Branch-stratified whole-day bootstrap does not model repeat-customer carryover or multi-day serial dependence",
                "Changing assignment probabilities requires a prespecified weighting analysis; no automatic IPW is applied",
                "Do not combine recommendation and layout gains or omit implementation costs",
                "Revenue already includes discounts; the discount column is audited, not subtracted twice"] ))
    return dict(status="Recorded pilot results; review design and uncertainty before rollout",experiments=reports,
        incremental_profit_measured=False,
        causal_status="Observed outcomes only; reporter cannot verify randomized assignment or identify a causal effect by itself")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--control",default="current")
    p.add_argument("--output")
    a=p.parse_args()
    try:
        result=summarize(pd.read_csv(a.csv),a.control)
    except (ValueError,KeyError) as exc:
        p.error(str(exc))
    text=json.dumps(result,indent=2,allow_nan=False)
    if a.output:
        Path(a.output).write_text(text,encoding="utf-8")
    else:
        print(text)


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
