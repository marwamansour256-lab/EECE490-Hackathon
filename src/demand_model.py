"""Leakage-aware one-day-ahead forecasts and an explicitly hypothetical stock policy.

Run from the repository root: python src/demand_model.py
No transaction rows are deduplicated. Forecasts update sales lags daily and refit
GLMs every seven days. April-May selects model/regularization; June is held out.
Actual target-day weather and retrospective conflict masks are oracle diagnostics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
START = pd.Timestamp("2026-01-01")
VALID_START = pd.Timestamp("2026-04-01")
TEST_START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-06-30")
ALPHAS = (0.1, 1.0, 10.0)
FAMILIES = ("internal", "calendar", "calendar_lag_weather", "oracle_actual_weather_conflict")
PERISHABLES = ("Bakery", "Sandwiches", "Breakfast", "Desserts")


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame({"Date": dates})
    # Calendar groups pool rare dates: six months cannot estimate each holiday.
    christian = ["01-06", "02-09", "03-25", "04-03", "04-05", "04-06", "04-10", "04-12", "04-13"]
    muslim = ["03-20", "03-21", "03-22", "03-23", "05-27", "05-28", "05-29", "06-16", "06-26"]
    civic = ["01-01", "05-01", "05-25"]
    md = out.Date.dt.strftime("%m-%d")
    out["christian_calendar"] = md.isin(christian).astype(float)
    out["muslim_calendar"] = md.isin(muslim).astype(float)
    out["civic_calendar"] = md.isin(civic).astype(float)
    out["ramadan"] = out.Date.between("2026-02-18", "2026-03-19").astype(float)
    # Gregorian dates are deterministic. Lunar closure windows require a dated
    # announcement available by the previous day's close, including eve flags.
    # PCM memos: Mar19 page26983, May18 page27162, Jun9 page27222,
    # Jun22 page27249; URLs and provenance are in external_evidence.json.
    known_at = {pd.Timestamp("2026-" + x): pd.Timestamp("2025-12-31") for x in christian + civic}
    known_at.update({pd.Timestamp("2026-" + x): pd.Timestamp("2026-03-19") for x in muslim if x.startswith("03-")})
    known_at.update({pd.Timestamp("2026-" + x): pd.Timestamp("2026-05-18") for x in muslim if x.startswith("05-")})
    known_at[pd.Timestamp("2026-06-16")] = pd.Timestamp("2026-06-09")
    known_at[pd.Timestamp("2026-06-26")] = pd.Timestamp("2026-06-22")
    for name, offset in (("holiday_eve", 1), ("holiday_after", -1)):
        out[name] = [float((event := target + pd.Timedelta(days=offset)) in known_at and known_at[event] <= target - pd.Timedelta(days=1)) for target in out.Date]
    # These are researcher-defined retrospective periods, not future warnings.
    out["retrospective_escalation"] = out.Date.between("2026-03-02", "2026-04-16").astype(float)
    out["retrospective_post_april17"] = (out.Date >= "2026-04-17").astype(float)
    payload = json.loads((ROOT / "data/external/beirut_weather_2026H1.json").read_text(encoding="utf-8"))
    weather = pd.DataFrame(payload["daily"]).rename(columns={"time": "Date", "temperature_2m_max": "temperature", "precipitation_sum": "rain", "wind_speed_10m_max": "wind"})
    weather["Date"] = pd.to_datetime(weather.Date)
    out = out.merge(weather[["Date", "temperature", "rain", "wind"]], on="Date", validate="one_to_one")
    out["rain"] = np.log1p(out.rain)
    for name in ("temperature", "rain", "wind"):
        out["previous_day_" + name] = out[name].shift(1)
    return out


def panel(raw: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    units = raw.groupby(["Date"] + dimensions, observed=True).Quantity.sum()
    # Product-category mapping is fixed; never invent invalid SKU-category pairs.
    groups = raw[dimensions].drop_duplicates().sort_values(dimensions)
    dates = pd.DataFrame({"Date": pd.date_range(START, END)})
    out = dates.merge(groups, how="cross").merge(units.rename("actual").reset_index(), how="left", on=["Date"] + dimensions)
    out["actual"] = out.actual.fillna(0).astype(float)
    out = out.sort_values(dimensions + ["Date"]).reset_index(drop=True)
    g = out.groupby(dimensions, observed=True, sort=False).actual
    out["last_weekday"] = g.shift(7)
    shifted = [g.shift(7 * k) for k in range(1, 5)]
    out["four_weekdays"] = pd.concat(shifted, axis=1).mean(axis=1)
    out["lag1"] = g.shift(1)
    out["roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    out["roll28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    out["history28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).sum())
    return out.sort_values(["Date"] + dimensions).reset_index(drop=True)


def feature_frame(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    d = d.copy()
    d["series"] = d.Branch + " / " + d.Category
    weekday = d.Date.dt.dayofweek.astype(str)
    d["category_weekday"] = d.Category + " / " + weekday
    d["branch_weekday"] = d.Branch + " / " + weekday
    d["trend"] = (d.Date - START).dt.days / 181.0
    base = ["trend"]
    for name in ("last_weekday", "four_weekdays", "lag1", "roll7", "roll28"):
        d["log_" + name] = np.log1p(d[name])
        base.append("log_" + name)
    for cat in sorted(d.Category.unique()):
        name = "trend_" + cat
        d[name] = d.trend * d.Category.eq(cat)
        base.append(name)
    cal = ["christian_calendar", "muslim_calendar", "civic_calendar", "ramadan", "holiday_eve", "holiday_after"]
    lag_weather = ["previous_day_temperature", "previous_day_rain", "previous_day_wind"]
    oracle = ["temperature", "rain", "wind", "retrospective_escalation", "retrospective_post_april17"]
    extras = {}
    interactions = {}
    for names, key in ((cal, "calendar"), (lag_weather, "weather"), (oracle, "oracle")):
        extras[key] = list(names)
        for cat in sorted(d.Category.unique()):
            for name in names:
                interaction = name + " / " + cat
                interactions[interaction] = d[name] * d.Category.eq(cat)
                extras[key].append(interaction)
    d = pd.concat([d, pd.DataFrame(interactions, index=d.index)], axis=1)
    columns = {
        "internal": base,
        "calendar": base + extras["calendar"],
        "calendar_lag_weather": base + extras["calendar"] + extras["weather"],
        "oracle_actual_weather_conflict": base + extras["calendar"] + extras["oracle"],
    }
    return d, columns


def forecast(d: pd.DataFrame, numeric: list[str], alpha: float) -> np.ndarray:
    prediction = np.full(len(d), np.nan)
    categorical = ["series", "category_weekday", "branch_weekday"]
    for refit in pd.date_range(VALID_START, END, freq="7D"):
        train = (d.Date >= START + pd.Timedelta(days=28)) & (d.Date < refit)
        target = (d.Date >= refit) & (d.Date < refit + pd.Timedelta(days=7))
        pre = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ])
        fit = make_pipeline(pre, PoissonRegressor(alpha=alpha, max_iter=500, tol=1e-6))
        fit.fit(d.loc[train, categorical + numeric], d.loc[train, "actual"])
        prediction[target] = fit.predict(d.loc[target, categorical + numeric])
    return prediction


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = np.asarray(predicted) - np.asarray(actual)
    return {"wape_pct": float(100 * np.abs(err).sum() / max(np.sum(actual), 1)), "mae_units": float(np.abs(err).mean()), "bias_pct": float(100 * err.sum() / max(np.sum(actual), 1))}


def add_metric_rows(rows: list[dict], d: pd.DataFrame, predictions: dict[str, np.ndarray], level: str) -> None:
    for period, use in (("validation_Apr_May", d.Date.between(VALID_START, TEST_START - pd.Timedelta(days=1))), ("test_June", d.Date.ge(TEST_START))):
        for name, pred in predictions.items():
            rows.append({"period": period, "level": level, "scope": "all", "method": name, **metrics(d.loc[use, "actual"].to_numpy(), pred[use])})
            dimensions = ("Branch", "Category", "Product") if "Product" in d else ("Branch", "Category")
            for dim in dimensions:
                for val in sorted(d[dim].unique()):
                    keep = use & d[dim].eq(val)
                    rows.append({"period": period, "level": level, "scope": dim + ":" + val, "method": name, **metrics(d.loc[keep, "actual"].to_numpy(), pred[keep])})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "data/stories_transactions.csv", parse_dates=["Date"])
    catalog_columns = ["Branch", "Category", "Product"]
    initial_catalog = set(raw.loc[raw.Date.lt(VALID_START), catalog_columns].itertuples(index=False, name=None))
    full_catalog = set(raw[catalog_columns].itertuples(index=False, name=None))
    assert full_catalog == initial_catalog, "Forecast catalog must be known before validation; new products need a separate cold-start rule."
    cats = panel(raw, ["Branch", "Category"]).merge(calendar_frame(pd.date_range(START, END)), on="Date", validate="many_to_one")
    cats, columns = feature_frame(cats)
    validation = cats.Date.between(VALID_START, TEST_START - pd.Timedelta(days=1))
    predictions = {"last_weekday": cats.last_weekday.to_numpy(), "four_weekdays": cats.four_weekdays.to_numpy()}
    tuning = []
    chosen_alphas = {}
    for family in FAMILIES:
        best = None
        for alpha in ALPHAS:
            pred = forecast(cats, columns[family], alpha)
            score = metrics(cats.loc[validation, "actual"].to_numpy(), pred[validation])
            tuning.append({"family": family, "alpha": alpha, **score})
            print(f"Validation {family}, alpha={alpha}: WAPE={score['wape_pct']:.3f}%", flush=True)
            if best is None or score["wape_pct"] < best[0]:
                best = (score["wape_pct"], alpha, pred)
        chosen_alphas[family] = best[1]
        predictions[family] = best[2]
    deployable = [x for x in predictions if not x.startswith("oracle")]
    selected_category = min(deployable, key=lambda x: metrics(cats.loc[validation, "actual"].to_numpy(), predictions[x][validation])["wape_pct"])
    metric_rows = []
    add_metric_rows(metric_rows, cats, predictions, "branch_category_day")
    cat_output = cats[["Date", "Branch", "Category", "actual"]].copy()
    for name, values in predictions.items():
        cat_output[name] = values
    cat_output = cat_output[cat_output.Date.ge(VALID_START)].copy()
    cat_output.to_csv(OUT / "model_predictions.csv", index=False)

    skus = panel(raw, ["Branch", "Category", "Product"])
    cat_history = skus.groupby(["Date", "Branch", "Category"], observed=True).history28.transform("sum")
    n_products = skus.groupby(["Date", "Branch", "Category"], observed=True).Product.transform("size")
    # Small symmetric Dirichlet pseudocount prevents a recent zero eliminating a SKU.
    skus["share"] = (skus.history28 + 1.0) / (cat_history + n_products)
    category_for_sku = cat_output.drop(columns="actual").rename(columns={name: "category_prediction_" + name for name in predictions})
    skus = skus.merge(category_for_sku, on=["Date", "Branch", "Category"], how="left", validate="many_to_one")
    sku_predictions = {"direct_sku_last_weekday": skus.last_weekday.to_numpy(), "direct_sku_four_weekdays": skus.four_weekdays.to_numpy()}
    for name in predictions:
        col = "category_prediction_" + name
        sku_predictions[name] = (skus[col] * skus.share).to_numpy()
    add_metric_rows(metric_rows, skus, sku_predictions, "branch_sku_day")
    sku_validation = skus.Date.between(VALID_START, TEST_START - pd.Timedelta(days=1))
    perish = skus.Category.isin(PERISHABLES)
    selection_mask = sku_validation & perish
    selected_sku = min([x for x in sku_predictions if not x.startswith("oracle")], key=lambda x: metrics(skus.loc[selection_mask, "actual"].to_numpy(), sku_predictions[x][selection_mask])["wape_pct"])

    # All financial constants come only from initial training, including discounts.
    train = raw[raw.Date.lt(VALID_START)].copy()
    train["total_cost"] = train["Unit Cost (USD)"] * train.Quantity
    prices = train.groupby(["Branch", "Product"]).agg(revenue=("Line Revenue (USD)", "sum"), total_cost=("total_cost", "sum"), units=("Quantity", "sum"))
    prices["price"] = prices.revenue / prices.units
    prices["cost"] = prices.total_cost / prices.units
    skus = skus.join(prices[["price", "cost"]], on=["Branch", "Product"], validate="many_to_one")
    assert skus[["price", "cost"]].notna().all().all(), "Unseen branch-SKU economics need a documented fallback."
    p, c, y = skus.price.to_numpy(), skus.cost.to_numpy(), skus.actual.to_numpy()
    finance_rows = []
    coverage = []
    for period, period_mask in (("validation_Apr_May", sku_validation), ("test_June", skus.Date.ge(TEST_START))):
        use = (period_mask & perish).to_numpy()
        for name, means in sku_predictions.items():
            means = np.nan_to_num(means, nan=0.0).clip(0, None)
            low, high = poisson.ppf(0.1, means), poisson.ppf(0.9, means)
            coverage.append({"period": period, "method": name, "nominal_central_coverage_pct": 80.0, "observed_coverage_pct": float(100 * ((y[use] >= low[use]) & (y[use] <= high[use])).mean()), "average_interval_width_units": float((high[use] - low[use]).mean())})
            for salvage_fraction in (0.0, 0.5):
                salvage = salvage_fraction * c
                fractile = np.clip((p - c) / (p - salvage), 0.001, 0.999)
                quantity = poisson.ppf(fractile, means)
                policies = {"cost_aware_poisson_quantile": quantity}
                if name == "direct_sku_four_weekdays":
                    policies["round_forecast_mean"] = np.rint(means)
                for policy, q in policies.items():
                    sold = np.minimum(q, y)
                    unsold = np.maximum(q - y, 0)
                    shortage = np.maximum(y - q, 0)
                    contrib = p * sold + salvage * unsold - c * q
                    for scope, scoped_mask in [("all_perishables", use)] + [("Branch:" + b, use & skus.Branch.eq(b).to_numpy()) for b in sorted(skus.Branch.unique())] + [("Category:" + cat, use & skus.Category.eq(cat).to_numpy()) for cat in PERISHABLES]:
                        finance_rows.append({"period": period, "method": name, "policy": policy, "salvage_fraction_of_cost": salvage_fraction, "scope": scope, "prepared_units": float(q[scoped_mask].sum()), "sold_units": float(sold[scoped_mask].sum()), "unsold_units": float(unsold[scoped_mask].sum()), "unmet_units": float(shortage[scoped_mask].sum()), "simulated_contribution_usd": float(contrib[scoped_mask].sum()), "perfect_information_contribution_usd": float(((p-c)*y)[scoped_mask].sum())})
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT / "model_metrics.csv", index=False)
    financial = pd.DataFrame(finance_rows)
    financial.to_csv(OUT / "preparation_simulation.csv", index=False)
    # Historical one-day-ahead decisions make the prototype concrete and auditable.
    recommendations = skus.loc[skus.Date.ge(TEST_START) & perish, ["Date", "Branch", "Category", "Product", "actual", "price", "cost"]].copy()
    rec_mask = (skus.Date.ge(TEST_START) & perish).to_numpy()
    recommendations["selected_method"] = selected_sku
    recommendations["predicted_mean_units"] = sku_predictions[selected_sku][rec_mask]
    recommendations["baseline_mean_units"] = sku_predictions["direct_sku_four_weekdays"][rec_mask]
    recommendations["zero_salvage_critical_fractile"] = (recommendations.price - recommendations.cost) / recommendations.price
    recommendations["recommended_prepare_units"] = poisson.ppf(recommendations.zero_salvage_critical_fractile, recommendations.predicted_mean_units).astype(int)
    recommendations["baseline_prepare_units"] = poisson.ppf(recommendations.zero_salvage_critical_fractile, recommendations.baseline_mean_units).astype(int)
    recommendations["demand_p10_units"] = poisson.ppf(0.1, recommendations.predicted_mean_units).astype(int)
    recommendations["demand_p90_units"] = poisson.ppf(0.9, recommendations.predicted_mean_units).astype(int)
    recommendations["status"] = "Historical backtest; Poisson interval uncalibrated; verify stocks and shelf life before live use"
    recommendations.to_csv(OUT / "preparation_recommendations.csv", index=False)
    for name, pred in predictions.items():
        assert np.isfinite(pred[cats.Date.ge(VALID_START)]).all() and (pred[cats.Date.ge(VALID_START)] >= 0).all(), name
    for name, pred in sku_predictions.items():
        assert np.isfinite(pred[skus.Date.ge(VALID_START)]).all() and (pred[skus.Date.ge(VALID_START)] >= 0).all(), name

    # Paired day bootstrap preserves correlation across branches/products on a date.
    test_cats = cats.Date.ge(TEST_START)
    incremental_ci = {}
    rng = np.random.default_rng(490)
    for challenger, comparator in (("calendar", "internal"), ("calendar_lag_weather", "internal"), ("oracle_actual_weather_conflict", "internal")):
        frame = pd.DataFrame({"Date": cats.loc[test_cats, "Date"], "difference": np.abs(predictions[challenger][test_cats] - cats.loc[test_cats, "actual"].to_numpy()) - np.abs(predictions[comparator][test_cats] - cats.loc[test_cats, "actual"].to_numpy())})
        daily = frame.groupby("Date").difference.sum().to_numpy()
        # 7-day moving blocks also retain short serial dependence (only 30 test days).
        n = len(daily)
        indices = rng.integers(0, n - 6, size=(2000, 5))[:, :, None] + np.arange(7)[None, None, :]
        samples = daily[indices.reshape(2000, -1)[:, :n]].mean(axis=1)
        incremental_ci[challenger + "_minus_" + comparator] = {"observed_daily_absolute_error_difference_units": float(daily.mean()), "moving_block_95pct_interval_units": [float(x) for x in np.quantile(samples, [0.025, 0.975])], "negative_means_improvement": True, "warning": "Only 30 test days; this uncertainty interval is descriptive and does not establish causality."}

    financial_ci = {}
    zero_salvage_fractile = np.clip((p-c)/p, 0.001, 0.999)
    def simulated_contribution(means: np.ndarray) -> np.ndarray:
        q = poisson.ppf(zero_salvage_fractile, np.nan_to_num(means, nan=0.0).clip(0, None))
        return p*np.minimum(q, y)-c*q
    selected_contribution = simulated_contribution(sku_predictions[selected_sku])
    finance_test = skus.Date.ge(TEST_START) & perish
    for comparator in ("four_weekdays", "direct_sku_four_weekdays"):
        difference = selected_contribution - simulated_contribution(sku_predictions[comparator])
        by_date = pd.DataFrame({"Date": skus.loc[finance_test, "Date"], "difference": difference[finance_test]}).groupby("Date").difference.sum().to_numpy()
        n = len(by_date)
        # Circular 7-day blocks keep each test day equally likely while preserving
        # most short-range dependence. No refitting or selection on test profits.
        indices = (rng.integers(0, n, size=(2000, 5))[:, :, None] + np.arange(7)[None, None, :]) % n
        sampled_total = by_date[indices.reshape(2000, -1)[:, :n]].sum(axis=1)
        financial_ci[selected_sku + "_minus_" + comparator] = {
            "june_simulated_contribution_difference_usd": float(by_date.sum()),
            "circular_7day_block_95pct_interval_total_usd": [float(x) for x in np.quantile(sampled_total, [0.025, 0.975])],
            "bootstrap_repetitions": 2000,
            "positive_means_improvement": True,
            "warning": "Exploratory 30-day uncertainty conditional on fixed models and assumed economics; not causal uplift, not uncertainty over stockouts/costs/capacity or future months.",
        }

    results = {
        "raw_rows_retained": len(raw),
        "selection": {"category_model_selected_on_Apr_May_WAPE": selected_category, "preparation_model_selected_on_Apr_May_perishable_SKU_WAPE": selected_sku, "selected_regularization_by_family": chosen_alphas, "hyperparameter_validation": tuning},
        "protocol": {"initial_training": "2026-01-01 through 2026-03-31 (first 28 days establish lags)", "validation": "2026-04-01 through 2026-05-31", "held_out_final_test": "2026-06-01 through 2026-06-30", "horizon": "one day ahead after previous-day sales and weather summaries are available", "refit": "expanding training set every 7 days beginning April 1; daily sales lags updated; no target-day sales in features", "selection": "regularization and winner chosen using April-May only; June metrics never select a model", "model": "L2-regularized pooled Poisson GLM, branch-category and weekday interactions, category linear trends, lag1/7 and rolling7/28 and four previous same-weekday sales", "sku_allocation": "category forecast multiplied by previous28-day SKU unit share with +1 pseudocount; direct SKU seasonal baselines also evaluated", "weather": "Beirut gridded proxy; branch locations unknown; previous-day realized/reanalysis values are NOT issued forecasts and archive revisions/availability are unverified", "oracle": "target-day actual weather and retrospective March2-Apr16 / post-April17 event periods; excluded from deployable model selection", "holiday_dates": "Ramadan Feb18-Mar19 (Feb17 declaration), Fitr Mar20-23 (Mar19 memo), Adha May27-29 (May18 memo), Hijri June16 (June9 memo), Ashura June26 (June22 memo); eve flags obey previous-day announcement cutoff", "holiday_availability_limit": "Announcement dates are verified, but intraday timestamps are not. Assumes announcements were available by local end of announcement day. Cafe closures are not implied."},
        "overall_metrics": metric_df[metric_df.scope.eq("all")].to_dict(orient="records"),
        "incremental_external_error_uncertainty": incremental_ci,
        "financial_difference_uncertainty_zero_salvage": financial_ci,
        "poisson_interval_diagnostics_perishables": coverage,
        "financial_results_all_perishables": financial[financial.scope.eq("all_perishables")].to_dict(orient="records"),
        "financial_assumptions": [
            "These are counterfactual scenario calculations, not measured savings or causal company uplift.",
            "Recorded units are treated as uncensored demand; actual stockouts, opening inventories, carryover and waste are unknown.",
            "Each SKU is prepared once before demand, with no replenishment, capacity constraints, substitution or unmet-demand carryover.",
            "Branch-SKU selling price and unit cost are quantity-weighted Jan-March revenue/cost per unit; labor, rent and delivery commissions are excluded.",
            "Demand is assumed Poisson around each predicted mean; quantile calibration is reported and this distribution is not guaranteed.",
            "The common newsvendor critical fractile is (price-cost)/(price-salvage); all compared forecasts use the SAME decision rule.",
            "Salvage scenarios equal 0% or 50% of unit cost per unsold unit; all prepared units incur unit cost.",
            "Bakery, Sandwiches, Breakfast and Desserts are assumed perishable planning categories; actual shelf lives and production batches are unavailable.",
            "Perfect information is an unattainable upper bound, not a proposed forecast.",
            "Religious calendar flags describe dates only and make no inference about customer religion.",
        ],
    }
    (OUT / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Selected category model:", selected_category, flush=True)
    print("Selected preparation model:", selected_sku, flush=True)
    print(metric_df[metric_df.period.eq("test_June") & metric_df.scope.eq("all")].to_string(index=False), flush=True)
    print(financial[financial.period.eq("test_June") & financial.scope.eq("all_perishables") & financial.salvage_fraction_of_cost.eq(0)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
