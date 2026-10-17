"""Exploratory external-context audit, not a causal attribution model.

Run: python src/external_analysis.py
Raw lines are preserved. Source/date provenance: reports/external_evidence.json.
Weather is one Beirut proxy grid; branch locations are unknown. All event
coefficients are retrospective associations. Forecast evaluation lives separately
in demand_model.py. This file never treats same-day weather as a prior forecast.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
FIG = OUT / "figures"
R = "Line Revenue (USD)"
GP = "Line Gross Profit (USD)"


def calendar(dates, ramadan_start="2026-02-18"):
    """Research calendar: observed dates, NOT an as-of event feed."""
    x = pd.DataFrame(index=dates)
    for d in range(1, 7):
        x[f"weekday_{d}"] = (dates.dayofweek == d).astype(float)
    x["trend"] = np.arange(len(dates)) / 30
    x["trend2"] = x["trend"] ** 2
    x["christian"] = dates.isin(pd.to_datetime([
        "2026-01-06", "2026-02-09", "2026-03-25", "2026-04-03",
        "2026-04-05", "2026-04-06", "2026-04-10", "2026-04-12", "2026-04-13",
    ])).astype(float)
    muslim = list(pd.date_range("2026-03-20", "2026-03-23"))
    muslim += list(pd.date_range("2026-05-27", "2026-05-29"))
    muslim += list(pd.to_datetime(["2026-06-16", "2026-06-26"]))
    x["muslim"] = dates.isin(muslim).astype(float)
    x["civic"] = dates.isin(pd.to_datetime([
        "2026-01-01", "2026-05-01", "2026-05-25",
    ])).astype(float)
    x["ramadan"] = ((dates >= ramadan_start) & (dates <= "2026-03-19")).astype(float)
    x["escalation_phase"] = ((dates >= "2026-03-02") & (dates <= "2026-04-16")).astype(float)
    x["post_truce_phase"] = (dates >= "2026-04-17").astype(float)
    x["april8_shock"] = (dates == "2026-04-08").astype(float)
    x["pay_window_hypothesis"] = ((dates.day <= 3) | (dates.day >= dates.days_in_month - 2)).astype(float)
    x["exam_window_proxy"] = ((dates >= "2026-05-04") & (dates <= "2026-05-17")).astype(float)
    x["world_cup_window"] = (dates >= "2026-06-11").astype(float)
    # Promotions are observable schedules in this file, not randomized treatments.
    x["promotion_window"] = (((dates.month == 3) | (dates.month == 5)) & (dates.day >= 12) & (dates.day <= 19)).astype(float)
    w = pd.DataFrame(json.loads((ROOT / "data/external/beirut_weather_2026H1.json").read_text())["daily"])
    w.index = pd.to_datetime(w.pop("time"))
    assert w.index.is_unique and dates.isin(w.index).all()
    x["temperature_5c"] = w["temperature_2m_max"] / 5
    x["rain_10mm"] = w["precipitation_sum"] / 10
    x["wind_10kmh"] = w["wind_speed_10m_max"] / 10
    assert not x.isna().any().any()
    return x.astype(float)


FACTORS = ["temperature_5c", "rain_10mm", "wind_10kmh", "ramadan", "christian", "muslim", "civic",
           "escalation_phase", "post_truce_phase", "april8_shock", "pay_window_hypothesis", "exam_window_proxy", "world_cup_window"]


def fit_series(y, x):
    # OLS on units keeps effects in units and avoids log retransformation claims.
    # HAC handles serial correlation within a series; BH accounts for search
    # across series/factors. These intervals remain exploratory, not causal.
    use = x.drop(columns=[c for c in x if x[c].nunique() <= 1])
    return sm.OLS(y.astype(float), sm.add_constant(use, has_constant="add")).fit(cov_type="HAC", cov_kwds={"maxlags":7})


def json_safe(value):
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/stories_transactions.csv", parse_dates=["Date"])
    dates = pd.date_range(df.Date.min(), df.Date.max(), freq="D", name="Date")
    x = calendar(dates)
    x.to_csv(ROOT / "data/external/context_calendar.csv", index=True)
    orders = df.groupby("Transaction ID", sort=False).agg(Date=("Date", "first"), Branch=("Branch", "first"),
        Time=("Time", "first"), revenue=(R, "sum"), profit=(GP, "sum"))
    series = {"All branches / all categories": df.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)}
    for branch, sub in df.groupby("Branch"):
        series[f"{branch} / all categories"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    for category, sub in df.groupby("Category"):
        series[f"All branches / {category}"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    for (branch, category), sub in df.groupby(["Branch", "Category"]):
        series[f"{branch} / {category}"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    # Item-level weather checks are a defined extension, not a search for the
    # most attractive narrative after inspecting each item's peaks.
    for product, sub in df.groupby("Product"):
        series[f"All branches / product: {product}"] = sub.groupby("Date").Quantity.sum().reindex(dates, fill_value=0)
    tests = []
    sensitivity = []
    for label, y in series.items():
        xs = x.copy()
        # Nuisance promotion control only where the observed campaign applied.
        label_branch, label_category = label.split(" / ", 1)
        if label_category.startswith("product: "):
            product = label_category.removeprefix("product: ")
            label_category = df.loc[df.Product.eq(product), "Category"].iloc[0]
        if label_branch not in ["Harbour", "All branches"] or label_category not in ["Cold Coffee", "Desserts", "all categories"]:
            xs = xs.drop(columns="promotion_window")
        fitted = fit_series(y, xs)
        ci = fitted.conf_int()
        selected_factors = FACTORS[:2] if " / product:" in label else FACTORS
        for feature in selected_factors:
            sparse = int((xs[feature] != 0).sum()) < 5
            tests.append(dict(series=label, feature=feature, coefficient_units=float(fitted.params[feature]),
                ci_low_units=float(ci.loc[feature, 0]) if not sparse else np.nan, ci_high_units=float(ci.loc[feature, 1]) if not sparse else np.nan,
                mean_daily_units=float(y.mean()), effect_pct_of_mean=float(100 * fitted.params[feature] / y.mean()),
                p_value=float(fitted.pvalues[feature]) if not sparse else np.nan, observations=len(y), feature_days=int((xs[feature] != 0).sum()),
                inferential_status="descriptive_only_less_than_5_exposed_days" if sparse else "exploratory"))
        if label.endswith("all categories"):
            for variant in ["linear_trend", "ramadan_feb19"]:
                xx = xs.drop(columns="trend2") if variant == "linear_trend" else xs.assign(ramadan=calendar(dates, "2026-02-19")["ramadan"])
                f = fit_series(y, xx)
                for factor in ["ramadan", "escalation_phase", "post_truce_phase", "christian", "muslim"]:
                    sensitivity.append(dict(series=label, variant=variant, feature=factor,
                        coefficient_units=float(f.params[factor]), p_value=float(f.pvalues[factor])))
    tests = pd.DataFrame(tests)
    eligible = tests.p_value.notna()
    tests.loc[eligible,"q_value_bh"] = multipletests(tests.loc[eligible,"p_value"], method="fdr_bh")[1]
    # More conservative family adjustment under arbitrary dependence.
    tests.loc[eligible,"q_value_by"] = multipletests(tests.loc[eligible,"p_value"], method="fdr_by")[1]
    tests.sort_values("p_value").to_csv(OUT / "external_associations.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "external_sensitivity.csv", index=False)

    # Fixed pre-cutoff rolling day-ahead order baseline for inspecting spikes.
    daily = orders.groupby("Date").size().reindex(dates, fill_value=0).rename("orders").to_frame()
    base_x = sm.add_constant(x[[c for c in x if c.startswith("weekday_")] + ["trend"]])
    daily["expected_prior_data"] = np.nan
    for i in range(42, len(dates)):
        f = sm.GLM(daily.orders.iloc[:i], base_x.iloc[:i], family=sm.families.Poisson()).fit()
        daily.iloc[i, daily.columns.get_loc("expected_prior_data")] = float(f.predict(base_x.iloc[[i]]).iloc[0])
    daily["residual_pct"] = 100 * (daily.orders / daily.expected_prior_data - 1)
    daily["weekday"] = dates.day_name()
    daily["temp_max_c_proxy"] = x.temperature_5c * 5
    daily["rain_mm_proxy"] = x.rain_10mm * 10
    daily["calendar_context"] = [", ".join(c for c in ["christian", "muslim", "civic", "ramadan", "april8_shock", "exam_window_proxy", "world_cup_window"] if x.loc[d, c]) or "No selected event flag" for d in dates]
    daily.to_csv(OUT / "daily_anomaly_audit.csv")
    tues = daily.loc[daily.weekday == "Tuesday"].dropna(subset=["expected_prior_data"])

    # Descriptive windows are NOT annual seasonal coefficients.
    df["season_window"] = np.select([df.Date.dt.month <= 2, df.Date.dt.month <= 5], ["winter_Jan-Feb", "spring_Mar-May"], default="early_summer_Jun")
    seas = df.groupby(["season_window", "Branch", "Category"]).agg(units=("Quantity", "sum"), revenue=(R, "sum"), profit=(GP, "sum"))
    daycounts = df.groupby("season_window").Date.nunique()
    seas["units_per_day"] = [u / daycounts[s] for (s, b, c), u in seas.units.items()]
    seas.to_csv(OUT / "season_category_summary.csv")
    items = df.groupby(["Branch", "Product", "Category"]).agg(units=("Quantity", "sum"), revenue=(R, "sum"), gross_profit=(GP, "sum"), price=("Unit Price (USD)", "first"), cost=("Unit Cost (USD)", "first"))
    items.to_csv(OUT / "product_economics.csv")
    # Ramadan hour-profile: fixed wall-clock daypart, not actual sunset/fasting.
    orders["hour"] = orders.Time.str[:2].astype(int)
    profile = []
    for branch in ["All branches", *sorted(df.Branch.unique())]:
        o = orders if branch == "All branches" else orders.loc[orders.Branch == branch]
        all_count = o.groupby("Date").size().reindex(dates, fill_value=0)
        day_count = o[o.hour.between(8, 16)].groupby("Date").size().reindex(dates, fill_value=0)
        share = day_count / all_count
        fit = fit_series(100 * share, x.drop(columns="promotion_window"))
        ci = fit.conf_int().loc["ramadan"]
        mask = x.ramadan.astype(bool)
        profile.append(dict(branch=branch, ramadan_daytime_share=float(day_count[mask].sum() / all_count[mask].sum()),
            other_daytime_share=float(day_count[~mask].sum() / all_count[~mask].sum()),
            adjusted_ramadan_percentage_points=float(fit.params.ramadan), ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]), p_value=float(fit.pvalues.ramadan)))
    prof = pd.DataFrame(profile)
    prof["q_value_bh"] = multipletests(prof.p_value, method="fdr_bh")[1]
    prof.to_csv(OUT / "ramadan_daypart.csv", index=False)

    # Diagnostic feature collinearity matters when overlapping regimes are used.
    z = x[FACTORS + ["trend", "trend2"]].corr()
    z.to_csv(OUT / "external_feature_correlations.csv")
    # Product-level thermal mix: explicit names, not whole Matcha & Tea category.
    df["is_iced_named"] = df.Product.str.contains("iced|cold brew|frappe", case=False, regex=True)
    mix = df.groupby(df.Date.dt.month).apply(lambda g: pd.Series({"units":g.Quantity.sum(),
        "iced_named_share":g.loc[g.is_iced_named, "Quantity"].sum()/g.Quantity.sum()}))
    mix.to_csv(OUT / "monthly_iced_mix.csv")
    dup = df.drop(columns=["season_window", "is_iced_named"]).duplicated()
    summary = dict(rows=len(df), orders=len(orders), units=int(df.Quantity.sum()), revenue=float(df[R].sum()),
        gross_profit=float(df[GP].sum()), excess_repeated_rows=int(dup.sum()), repeated_revenue=float(df.loc[dup,R].sum()),
        repeated_gross_profit=float(df.loc[dup, GP].sum()), branches=sorted(df.Branch.unique()),
        periods=len(dates), association_count=len(tests), test_count=int(eligible.sum()), series_count=len(series),
        bh_discoveries=int((tests.q_value_bh < .05).sum()), by_discoveries=int((tests.q_value_by < .05).sum()),
        network_associations=tests.loc[tests.series == "All branches / all categories"].to_dict("records"),
        discoveries=tests.loc[tests.q_value_bh < .05].sort_values("q_value_bh").to_dict("records"),
        busiest_tuesday=tues.sort_values("orders", ascending=False).head(1).reset_index().assign(Date=lambda d:d.Date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        most_unusual_tuesday=tues.sort_values("residual_pct", ascending=False).head(1).reset_index().assign(Date=lambda d:d.Date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        largest_positive_dates=daily.dropna().nlargest(5, "residual_pct").reset_index().assign(Date=lambda d:d.Date.dt.strftime("%Y-%m-%d")).to_dict("records"),
        ramadan_dayparts=prof.to_dict("records"),
        notes=["OLS in units with weekday, linear/quadratic trend and joint external factors; Newey-West HAC lag7.",
            "Exploratory search family: network/branch/category context plus 45 network product weather checks. BH and more conservative BY reported. Neither proves causality; event reuse and sparse exposures limit precision.",
            "Daypart family has four additional tests, adjusted separately. 08:00-16:59 is a clock proxy, not religious observance.",
            "Christian/Muslim label calendar events, never customer religion. Public closures do not establish cafe hours.",
            "Single April8 shock coefficient is descriptive; its robust interval is not credible replication of a one-day event.",
            "Exam and World Cup windows are exploratory national proxies: no verified campus catchment or screening.",
            "Weather is the cached Beirut grid; no verified mapping to branches. Seasons are descriptive windows only."])
    (OUT / "external_analysis_results.json").write_text(json.dumps(json_safe(summary), indent=2, default=str, allow_nan=False), encoding="utf-8")

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios":[2,1]}, sharex=True)
    axes[0].plot(daily.index, daily.orders, color="#b5bec9", lw=1, label="Observed orders")
    axes[0].plot(daily.index, daily.expected_prior_data, color="#176b70", lw=1.6, label="Forecast using only earlier sales")
    axes[0].set_title("A busy Tuesday is not automatically an event-driven Tuesday", loc="left", weight="bold")
    axes[0].set_ylabel("Orders / day")
    axes[0].legend(frameon=False)
    axes[1].bar(daily.index, daily.residual_pct, width=1, color="#176b70")
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].set_ylabel("Actual vs forecast (%)")
    for d,label in [("2026-03-02","Escalation"),("2026-04-08","April 8 strikes"),("2026-04-17","Temporary truce")]:
        axes[0].axvline(pd.Timestamp(d),color="#c66c38",lw=.8,ls="--")
        axes[0].annotate(label,(pd.Timestamp(d),daily.orders.max()),rotation=90,va="top",fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "context_forecast_audit.png", dpi=160)
    plt.close(fig)
    network = tests[tests.series == "All branches / all categories"].set_index("feature").loc[FACTORS[:9]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(network.coefficient_units, np.arange(len(network)),
        xerr=np.vstack([network.coefficient_units-network.ci_low_units,network.ci_high_units-network.coefficient_units]),
        fmt="o",color="#176b70",capsize=3)
    ax.axvline(0,color="#7d838a",lw=.8)
    ax.set_yticks(np.arange(len(network)),network.index)
    ax.set_xlabel("Adjusted daily units (weather per 5C / 10mm / 10kmh); 95% HAC interval")
    ax.set_title("External associations are estimates with uncertainty",loc="left",weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "context_effect_uncertainty.png",dpi=160)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ["rows","orders","test_count","bh_discoveries","by_discoveries","busiest_tuesday","most_unusual_tuesday"]},indent=2))


if __name__ == "__main__":
    main()
