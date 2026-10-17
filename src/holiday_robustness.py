"""Post-selection sensitivity audit of the Sandwiches holiday association.

Run: .venv/Scripts/python.exe src/holiday_robustness.py
This reuses the discovery data; it is neither causal identification nor new
independent validation. Four event blocks are not nine independent holidays.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from external_analysis import calendar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
EVENTS = {
    "Eid al-Fitr": pd.date_range("2026-03-20", "2026-03-23"),
    "Eid al-Adha": pd.date_range("2026-05-27", "2026-05-29"),
    "Hijri New Year": pd.to_datetime(["2026-06-16"]),
    "Ashoura": pd.to_datetime(["2026-06-26"]),
}


def fit(y, x):
    # Drop constants in reduced samples; retain explicit intercept.
    xx = x.loc[:, x.nunique() > 1]
    return sm.OLS(y.astype(float), sm.add_constant(xx, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})


def predict(f, x):
    xx = sm.add_constant(x, has_constant="add")
    return f.predict(xx[f.model.exog_names])


def estimate(f, feature="muslim"):
    ci = f.conf_int().loc[feature]
    return {"coefficient_units_per_day": float(f.params[feature]),
            "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
            "p_value_unadjusted": float(f.pvalues[feature])}


def main():
    df = pd.read_csv(ROOT / "data/stories_transactions.csv", parse_dates=["Date"])
    dates = pd.date_range(df.Date.min(), df.Date.max(), freq="D")
    x = calendar(dates)
    base_cols = [c for c in x if c.startswith("weekday_")] + ["trend"]
    specs = {
        "weekday_linear": x[base_cols + ["muslim", "christian"]].copy(),
        "weekday_quadratic": x[base_cols + ["trend2", "muslim", "christian"]].copy(),
        "full_context_linear": x.drop(columns=["trend2", "promotion_window"]).copy(),
        "full_context_quadratic": x.drop(columns="promotion_window").copy(),
    }
    sandwich = df[df.Category.eq("Sandwiches")]
    series = {"All branches / Sandwiches": sandwich.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)}
    for branch, sub in sandwich.groupby("Branch"):
        series[f"{branch} / Sandwiches"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    for product, sub in sandwich.groupby("Product"):
        series[f"All branches / {product}"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    y = series["All branches / Sandwiches"]
    all_mask = x.muslim.eq(1)
    assert all_mask.sum() == 9
    rows, model_sensitivity, subgroup, omitted, off_event, daily_off, matched = [], [], [], [], [], [], []

    for name, xx in specs.items():
        f = fit(y, xx)
        rec = {"analysis": "model_sensitivity", "series": "All branches / Sandwiches", "specification": name,
               "n_days": len(y), "event_days": 9, **estimate(f)}
        model_sensitivity.append(rec)
        rows.append(rec)
        # Remove a whole event from the fitting sample, preserving its dates in
        # the prediction frame. The remaining events retain their true flags.
        for event, event_dates in EVENTS.items():
            keep = ~dates.isin(event_dates)
            ff = fit(y.loc[keep], xx.loc[keep])
            pred = predict(ff, xx.loc[~keep])
            pred_no_event = predict(ff, xx.loc[~keep].assign(muslim=0))
            rec = {"analysis": "omit_entire_event", "series": "All branches / Sandwiches", "specification": name,
                   "event": event, "n_days": int(keep.sum()), "event_days": int(xx.loc[keep, "muslim"].sum()),
                   **estimate(ff), "omitted_actual_daily": float(y.loc[~keep].mean()),
                   "omitted_baseline_daily": float(pred_no_event.mean()),
                   "omitted_observed_minus_baseline": float((y.loc[~keep] - pred_no_event).mean()),
                   "omitted_prediction_error_with_effect": float((y.loc[~keep] - pred).mean())}
            omitted.append(rec)
            rows.append(rec)
        # Fit only dates outside every Muslim event and predict all nine.
        nuisance = xx.drop(columns="muslim")
        ff = fit(y.loc[~all_mask], nuisance.loc[~all_mask])
        pred = predict(ff, nuisance.loc[all_mask])
        for event, event_dates in EVENTS.items():
            actual = y.reindex(event_dates)
            expected = pred.reindex(event_dates)
            rec = {"analysis": "fit_off_all_muslim_events", "series": "All branches / Sandwiches", "specification": name,
                   "event": event, "n_days": int((~all_mask).sum()), "event_days": len(event_dates),
                   "actual_units": float(actual.sum()), "expected_units": float(expected.sum()),
                   "coefficient_units_per_day": float((actual - expected).mean()),
                   "deviation_pct_of_expected": float(100 * (actual.sum() / expected.sum() - 1))}
            off_event.append(rec)
            rows.append(rec)
        for day in pred.index:
            daily_off.append({"specification": name, "date": day.strftime("%Y-%m-%d"),
                              "actual_units": int(y.loc[day]), "expected_units": float(pred.loc[day]),
                              "deviation_units": float(y.loc[day] - pred.loc[day])})

    for label, yy in series.items():
        f = fit(yy, specs["full_context_quadratic"])
        rec = {"analysis": "branch_or_item", "series": label, "specification": "full_context_quadratic",
               "mean_daily_units": float(yy.mean()), **estimate(f)}
        rec["effect_pct_of_mean"] = 100 * rec["coefficient_units_per_day"] / rec["mean_daily_units"]
        subgroup.append(rec)
        rows.append(rec)

    # Calendar definitions differ in coverage, not the underlying event dates.
    definitions = {
        "public_closure_9days": [d for ds in EVENTS.values() for d in ds],
        "first_religious_day_4days": pd.to_datetime(["2026-03-20", "2026-05-27", "2026-06-16", "2026-06-26"]),
        "eid_two_days_6days": pd.to_datetime(["2026-03-20", "2026-03-21", "2026-05-27", "2026-05-28", "2026-06-16", "2026-06-26"]),
    }
    calendar_sensitivity = []
    for name, days in definitions.items():
        for spec in ("weekday_linear", "weekday_quadratic", "full_context_quadratic"):
            xx = specs[spec].assign(muslim=dates.isin(days).astype(float))
            f = fit(y, xx)
            rec = {"analysis": "holiday_definition", "series": "All branches / Sandwiches", "definition": name,
                   "specification": spec, "event_days": len(days), **estimate(f)}
            calendar_sensitivity.append(rec)
            rows.append(rec)

    # Local comparisons use BOTH earlier and later matching weekdays: explicitly
    # retrospective. A local linear slope adjusts for growth around the target.
    for day in dates[all_mask]:
        offset = np.asarray((dates - day).days)
        eligible = ((np.abs(offset) <= 28) & (offset != 0) & (dates.dayofweek == day.dayofweek)
                    & ~all_mask.to_numpy() & x.christian.eq(0).to_numpy() & x.civic.eq(0).to_numpy())
        local_x = pd.DataFrame({"offset_days": offset[eligible]}, index=dates[eligible])
        ff = sm.OLS(y.loc[eligible], sm.add_constant(local_x)).fit()
        expected = float(ff.params["const"])
        rec = {"analysis": "weekday_matched_local_linear_28days", "series": "All branches / Sandwiches",
               "date": day.strftime("%Y-%m-%d"), "n_controls": int(eligible.sum()),
               "earlier_controls": int(((offset < 0) & eligible).sum()),
               "later_controls": int(((offset > 0) & eligible).sum()),
               "actual_units": int(y.loc[day]), "expected_units": expected,
               "coefficient_units_per_day": float(y.loc[day] - expected)}
        matched.append(rec)
        rows.append(rec)

    # Explicit same-model contrast: beta_Muslim minus beta_Christian.
    contrasts = []
    all_y = df.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    for label, yy in [("All branches / Sandwiches", y), ("All branches / all categories", all_y)]:
        for name, xx in specs.items():
            if label.endswith("all categories") and name.startswith("full_context"):
                xx = xx.assign(promotion_window=x.promotion_window)
            f = fit(yy, xx)
            test = f.t_test("muslim - christian = 0")
            ci = np.asarray(test.conf_int()).reshape(-1)
            rec = {"analysis": "muslim_minus_christian_contrast", "series": label, "specification": name,
                   "coefficient_units_per_day": float(np.asarray(test.effect).item()),
                   "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                   "p_value_unadjusted": float(np.asarray(test.pvalue).item()),
                   "muslim_coefficient": float(f.params.muslim), "christian_coefficient": float(f.params.christian)}
            contrasts.append(rec)
            rows.append(rec)

    selected_events = [r for r in off_event if r["specification"] == "full_context_quadratic"]
    event_deviations = np.array([r["coefficient_units_per_day"] for r in selected_events])
    event_mean = float(event_deviations.mean())
    se = float(stats.sem(event_deviations))
    event_t = stats.t.interval(.95, df=3, loc=event_mean, scale=se)
    negative_events = int((event_deviations < 0).sum())
    weighted_cost = float((sandwich.Quantity * sandwich["Unit Cost (USD)"]).sum() / sandwich.Quantity.sum())
    effect = float(fit(y, specs["full_context_quadratic"]).params.muslim)
    theoretical_units = max(0, -effect) * 9
    summary = {
        "status": "POST_SELECTION_EXPLORATORY_SENSITIVITY_NOT_NEW_INDEPENDENT_VALIDATION",
        "raw_lines_preserved": True, "dates": len(dates), "muslim_event_blocks": 4, "muslim_exposed_days": 9,
        "model_sensitivity": model_sensitivity, "omitted_entire_event": omitted,
        "off_all_muslim_events": off_event, "off_event_daily_predictions": daily_off,
        "branch_and_item_effects": subgroup, "calendar_definition_sensitivity": calendar_sensitivity,
        "weekday_matched": matched,
        "weekday_matched_summary": {"mean_daily_deviation_units": float(np.mean([r["coefficient_units_per_day"] for r in matched])),
                                    "total_deviation_units": float(np.sum([r["coefficient_units_per_day"] for r in matched]))},
        "formal_religious_calendar_contrasts": contrasts,
        "four_event_descriptive_uncertainty": {
            "definition": "Equal event-weight off-event residuals, full-context quadratic. Heuristic only: four blocks and shared baseline; not an independent test.",
            "negative_events": negative_events, "event_mean_daily_deviation": event_mean,
            "t3_interval_low": float(event_t[0]), "t3_interval_high": float(event_t[1]),
            "two_sided_sign_test_p_unadjusted": float(stats.binomtest(negative_events, 4, .5).pvalue)},
        "theoretical_waste_scenario": {
            "weighted_sandwich_cost_usd": weighted_cost, "estimated_fewer_sold_units_per_holiday_day": -effect,
            "nine_day_units": theoretical_units, "nine_day_ingredient_cost_usd": theoretical_units * weighted_cost,
            "rounded_4_units_times_9_cost_usd": 4 * 9 * weighted_cost,
            "assumptions": "Every forecast reduction would otherwise be prepared and discarded, no salvage, no stockout/sales penalty, historical product/branch cost mix applies. No waste, inventory or censoring data: these are scenario dollars, not measured savings or annual ROI."},
        "limitations": [
            "Hypothesis was chosen from the full January-June discovery search, including June. All sensitivity analyses reuse those observations.",
            "HAC lag7 intervals account approximately for time dependence, not selection or reliable inference from four holiday blocks; subgroup/calendar p-values are unadjusted exploratory diagnostics.",
            "All coefficients are conditional associations. Public closure may change working schedules, opening hours, stock availability, promotions, travel or product substitution.",
            "The four event blocks occur once and overlap different regimes. Christian and Muslim are calendar categories, never observed or inferred customer religions.",
            "Omitted-event refits remove every row in that event; local comparisons and off-event fits use future observations and are retrospective sensitivity checks, not forecasting validation.",
            "Local matching has fewer later controls near the June dataset boundary; results cannot establish repeatable annual seasonality.",
            "Demand is censored by availability: fewer recorded sales could reflect fewer sandwiches offered. Do not reduce production operationally without stockout and waste logs."
        ]}
    (OUT / "holiday_robustness.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "holiday_robustness.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["model_sensitivity", "weekday_matched_summary", "four_event_descriptive_uncertainty", "theoretical_waste_scenario"]}, indent=2))


if __name__ == "__main__":
    main()
