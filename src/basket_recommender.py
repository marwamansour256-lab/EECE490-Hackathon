"""Frozen Stories basket-completion model and offline recommendation contract.

Historical co-purchases estimate relevance, not causal offer acceptance. Settings
are selected on March 11-31, refitted through March 31, then evaluated Apr-Jun.
No customer identifiers are included in the exported model or browser assets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = pd.Timestamp("2026-03-31")
VALID_START = pd.Timestamp("2026-03-11")
DAYPARTS = ["all", "morning", "midday", "evening"]
REGULAR_DATES = 6
CONTEXT_PRIOR = 200.0


def daypart(time):
    hour = int(str(time).split(":")[0])
    return "morning" if hour < 12 else "midday" if hour < 17 else "evening"


def load_orders(path):
    raw = pd.read_csv(path, parse_dates=["Date"], dtype={"Customer ID": "string"})
    for column in ["Date", "Branch", "Time", "Customer ID"]:
        if raw.groupby("Transaction ID")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Transaction contains inconsistent {column}")
    orders = raw.groupby("Transaction ID", sort=False).agg(
        date=("Date", "first"), branch=("Branch", "first"), time=("Time", "first"),
        customer=("Customer ID", "first"), products=("Product", lambda x: tuple(sorted(set(x)))),
        revenue=("Line Revenue (USD)", "sum"), gross_profit=("Line Gross Profit (USD)", "sum"),
    ).reset_index()
    orders["daypart"] = orders.time.map(daypart)
    return raw, orders


def regular_ids(orders, threshold=REGULAR_DATES):
    visits = orders.dropna(subset=["customer"]).groupby("customer").date.nunique()
    return set(visits[visits.ge(threshold)].index)


def stats(orders, names):
    ix = {p: i for i, p in enumerate(names)}
    single = np.zeros(len(names), dtype=float)
    pair = np.zeros((len(names), len(names)), dtype=float)
    for basket in orders.products:
        ids = [ix[p] for p in basket if p in ix]
        single[ids] += 1
        for a, b in itertools.combinations(ids, 2):
            pair[a, b] += 1
            pair[b, a] += 1
    return dict(n=len(orders), single=single, pair=pair)


def fit(raw, orders, method="shrunk_pairs", strength=100.0):
    """Fit only the explicitly supplied training rows, never global raw/test data."""
    if orders.empty:
        raise ValueError("Training orders must be nonempty")
    if orders.date.max() > TRAIN_END or raw.Date.max() > TRAIN_END:
        raise ValueError("Frozen recommendation fitting cannot include April-June sales")
    if raw.Date.max() != orders.date.max() or raw.Date.min() != orders.date.min():
        raise ValueError("Raw training economics and order dates must agree")
    catalog = raw[["Product", "Category"]].drop_duplicates().sort_values("Product")
    if catalog.Product.duplicated().any():
        raise ValueError("A product has multiple categories")
    names = catalog.Product.tolist()
    regular = regular_ids(orders)
    global_stats = stats(orders, names)
    contexts = {}
    economics = {}
    for branch in sorted(orders.branch.unique()):
        r = raw[raw.Branch.eq(branch)]
        econ = r.groupby("Product").agg(units=("Quantity", "sum"),
            margin=("Line Gross Profit (USD)", "sum"), price=("Line Revenue (USD)", "sum"))
        economics[branch] = {p: dict(margin=float(econ.loc[p, "margin"] / econ.loc[p, "units"]),
                                   price=float(econ.loc[p, "price"] / econ.loc[p, "units"]))
                             for p in names if p in econ.index}
        for part in DAYPARTS:
            for cohort in ["all", "regular"]:
                mask = orders.branch.eq(branch)
                if part != "all":
                    mask &= orders.daypart.eq(part)
                if cohort == "regular":
                    mask &= orders.customer.isin(regular)
                contexts[f"{branch}|{part}|{cohort}"] = stats(orders[mask], names)
    return dict(names=names, categories=dict(zip(catalog.Product, catalog.Category)),
        branches=sorted(orders.branch.unique()), global_stats=global_stats, contexts=contexts,
        economics=economics, method=method, strength=float(strength), regular=regular,
        trained_through=str(orders.date.max().date()))


def context_arrays(model, key):
    g = model["global_stats"]
    c = model["contexts"][key]
    p = (g["single"] + 0.5) / (g["n"] + 1.0)
    base = (c["single"] + CONTEXT_PRIOR * p) / (c["n"] + CONTEXT_PRIOR)
    alpha = model["strength"]
    global_cond = (g["pair"] + alpha * p[None, :]) / (g["single"][:, None] + alpha)
    conditional = (c["pair"] + alpha * global_cond) / (c["single"][:, None] + alpha)
    margins = np.array([model["economics"][key.split("|")[0]].get(n, {}).get("margin", 0)
                        for n in model["names"]])
    method = model["method"]
    prob = conditional if method == "shrunk_pairs" else np.broadcast_to(base, conditional.shape)
    scores = prob if method == "popularity" else prob * np.maximum(margins[None, :], 0)
    baseline_scores = base if method == "popularity" else base * np.maximum(margins, 0)
    return base, prob, scores, baseline_scores, margins


def rank_indices(names, scores, excluded=()):
    excluded = set(excluded)
    return sorted((i for i, p in enumerate(names) if p not in excluded),
                  key=lambda i: (-float(scores[i]), names[i]))


def score_basket(model, branch, part, cohort, basket):
    key = f"{branch}|{part}|{cohort}"
    if key not in model["contexts"]:
        key = f"{branch}|all|all"
    base, prob, scores, baseline, margins = context_arrays(model, key)
    index = {p: i for i, p in enumerate(model["names"])}
    anchors = [index[p] for p in basket if p in index]
    return (scores[anchors].max(axis=0) if anchors else baseline), margins


def evaluate(model, orders, label):
    """Each multi-product order contributes equal weight across its hidden items."""
    rows = []
    names = model["names"]
    ix = {p: i for i, p in enumerate(names)}
    # Cache array/rank per context+observed basket; no test target enters fitting.
    cache = {}
    for order in orders.itertuples(index=False):
        if len(order.products) < 2:
            continue
        cohort = "regular" if order.customer in model["regular"] else "all"
        observed = order.products
        accum = np.zeros(4)
        unknown = 0
        for target in observed:
            basket = tuple(p for p in observed if p != target)
            key = (order.branch, order.daypart, cohort, basket)
            if key not in cache:
                scores, margins = score_basket(model, *key)
                ranking = rank_indices(names, scores, basket)
                cache[key] = ({names[i]: pos + 1 for pos, i in enumerate(ranking)}, margins)
            ranks, margins = cache[key]
            rank = ranks.get(target, len(names) + 1)
            unknown += target not in ix
            margin = margins[ix[target]] if target in ix else 0
            accum += [rank <= 3, 1 / rank if target in ranks else 0,
                      margin * (rank <= 3), rank <= 1]
        accum /= len(observed)
        rows.append(dict(date=str(order.date.date()), branch=order.branch,
            cohort="unidentified" if pd.isna(order.customer) else "regular" if cohort == "regular" else "other_identified",
            hit_at_3=accum[0], reciprocal_rank=accum[1], recovered_margin_at_3_usd=accum[2],
            hit_at_1=accum[3], unknown_targets=unknown, targets=len(observed)))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No multi-product baskets to evaluate")
    metrics = ["hit_at_3", "reciprocal_rank", "recovered_margin_at_3_usd", "hit_at_1"]
    summary = dict(label=label, method=model["method"], strength=model["strength"],
        orders=len(frame), dates=int(frame.date.nunique()), masked_items=int(frame.targets.sum()),
        unknown_targets=int(frame.unknown_targets.sum()), **frame[metrics].mean().to_dict())
    summary["cohorts"] = {k: dict(orders=len(x), **x[metrics].mean().to_dict())
                          for k, x in frame.groupby("cohort")}
    return summary, frame


def export_model(model):
    payload = dict(schema_version=1, status="Historical model; prospective sales uplift unmeasured",
        trained_through=model["trained_through"], selected_method=model["method"],
        shrinkage_strength=model["strength"], branches=model["branches"], dayparts=DAYPARTS,
        cohorts=["all", "regular"], regular_definition=f">={REGULAR_DATES} distinct training purchase dates",
        products=[dict(product=p, category=model["categories"][p]) for p in model["names"]], contexts={})
    for key, c in model["contexts"].items():
        base, prob, scores, baseline_scores, margins = context_arrays(model, key)
        branch = key.split("|")[0]

        def row(j, i=None):
            probability = base[j] if i is None else prob[i, j]
            pair_orders = 0 if i is None else int(c["pair"][i, j])
            return dict(product=model["names"][j], category=model["categories"][model["names"][j]],
                score=round(float(baseline_scores[j] if i is None else scores[i, j]), 10),
                conditional_probability=round(float(probability), 8), baseline_probability=round(float(base[j]), 8),
                lift=round(float(probability / base[j]), 6), pair_orders=pair_orders,
                anchor_orders=0 if i is None else int(c["single"][i]),
                unit_margin_usd=round(float(margins[j]), 6),
                unit_price_usd=round(model["economics"][branch].get(model["names"][j], {}).get("price", 0), 6),
                reason="Shrunk historical co-purchase and unit margin" if i is not None and model["method"] == "shrunk_pairs"
                    else "Training popularity" + (" and unit margin" if model["method"] != "popularity" else ""),
                evidence_status="Association, not incremental acceptance")

        payload["contexts"][key] = dict(training_orders=c["n"],
            fallback=f"Context pooled toward all training orders; {CONTEXT_PRIOR:g}-order popularity prior and {model['strength']:g}-anchor pair prior",
            baseline=[row(j) for j in rank_indices(model["names"], baseline_scores)],
            by_anchor={p: [row(j, i) for j in rank_indices(model["names"], scores[i], [p])]
                       for i, p in enumerate(model["names"])})
    return payload


def recommend(payload, branch, basket=(), part="all", cohort="all", unavailable=(), limit=3):
    if branch not in payload["branches"]:
        raise ValueError(f"Unknown branch {branch}")
    key = f"{branch}|{part}|{cohort}"
    fallback = key not in payload["contexts"]
    if fallback:
        key = f"{branch}|all|all"
    context = payload["contexts"][key]
    lists = [context["by_anchor"].get(p, context["baseline"]) for p in basket] if basket else [context["baseline"]]
    excluded = set(basket) | set(unavailable)
    best = {}
    for candidates in lists:
        for candidate in candidates:
            product = candidate["product"]
            if product not in excluded and (product not in best or candidate["score"] > best[product]["score"]):
                best[product] = candidate
    ranked = sorted(best.values(), key=lambda r: (-r["score"], r["product"]))[:limit]
    return dict(status=payload["status"], trained_through=payload["trained_through"],
        method=payload["selected_method"], context=key, context_fallback=fallback,
        pooling=context["fallback"], basket=list(basket), recommendations=ranked,
        money_interpretation="Score ranks historical relevance/margin, not incremental profit or acceptance probability")


def paired_interval(a, b, seed=490):
    """Paired seven-consecutive-date moving-block bootstrap; dates stay together."""
    x = a.assign(delta=a.recovered_margin_at_3_usd.to_numpy() - b.recovered_margin_at_3_usd.to_numpy())
    days = pd.date_range(x.date.min(), x.date.max()).strftime("%Y-%m-%d")
    groups = x.groupby("date").delta.agg(["sum", "count"]).reindex(days, fill_value=0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        starts = rng.integers(0, len(days) - 6, size=int(np.ceil(len(days) / 7)))
        indices = np.concatenate([np.arange(s, s + 7) for s in starts])[:len(days)]
        selected = groups.iloc[indices]
        samples.append(selected["sum"].sum() / selected["count"].sum())
    return dict(mean_delta_usd_per_completion_order=float(x.delta.mean()),
        conditional_95pct_interval=np.quantile(samples, [.025, .975]).tolist(),
        interpretation="Offline recovered-item margin proxy only; not incremental sales or randomized treatment evidence")


def train(path):
    raw, orders = load_orders(path)
    train_orders = orders[orders.date.le(TRAIN_END)]
    train_raw = raw[raw.Date.le(TRAIN_END)]
    inner_o = train_orders[train_orders.date.lt(VALID_START)]
    inner_r = train_raw[train_raw.Date.lt(VALID_START)]
    validation = train_orders[train_orders.date.ge(VALID_START)]
    candidates = [("popularity", 100), ("margin_popularity", 100)] + [("shrunk_pairs", a) for a in [20, 100, 500]]
    validation_results = []
    for method, strength in candidates:
        model = fit(inner_r, inner_o, method, strength)
        result, _ = evaluate(model, validation, "training_validation")
        validation_results.append(result)
        print(json.dumps({k: result[k] for k in ["method", "strength", "hit_at_3", "recovered_margin_at_3_usd"]}), flush=True)
    # Ranking objective is declared before test evaluation. Baselines win exact ties.
    chosen = max(range(len(candidates)), key=lambda i: validation_results[i]["recovered_margin_at_3_usd"])
    selected = candidates[chosen]
    final = fit(train_raw, train_orders, *selected)
    payload = export_model(final)
    test = orders[orders.date.gt(TRAIN_END)]
    test_summaries, test_frames = [], {}
    # Report every predeclared candidate, never replace the selected winner on test.
    for method, strength in candidates:
        candidate = dict(final, method=method, strength=float(strength))
        summary, frame = evaluate(candidate, test, "chronological_test")
        name = f"{method}:{strength}"
        test_summaries.append(summary)
        test_frames[name] = frame
    selected_name = f"{selected[0]}:{selected[1]}"
    regular = final["regular"]
    def population(o):
        identified = o.customer.notna()
        loyal = o.customer.isin(regular)
        return dict(orders=len(o), identified_orders=int(identified.sum()),
            regular_orders=int(loyal.sum()), regular_revenue_share_all_orders=float(o.loc[loyal, "revenue"].sum()/o.revenue.sum()),
            regular_revenue_share_identified=float(o.loc[loyal, "revenue"].sum()/o.loc[identified, "revenue"].sum()),
            unidentified_order_share=float((~identified).mean()),
            single_distinct_product_order_share=float(o.products.map(len).eq(1).mean()))
    report = dict(schema_version=1, model="Smoothed conditional basket-frequency model with margin-aware ranking and popularity fallback",
        train_dates=[str(train_orders.date.min().date()), str(TRAIN_END.date())],
        test_dates=[str(test.date.min().date()), str(test.date.max().date())],
        validation_dates=[str(VALID_START.date()), str(TRAIN_END.date())],
        selection_objective="Mean recovered hidden-item training gross margin at rank<=3, equal weight per multi-item order",
        selected_method=selected[0], selected_strength=selected[1], validation=validation_results, test=test_summaries,
        selected_vs_margin_baseline=paired_interval(test_frames[selected_name], test_frames["margin_popularity:100"]),
        selected_vs_popularity=paired_interval(test_frames[selected_name], test_frames["popularity:100"]),
        population={"train": population(train_orders), "test": population(test)},
        regular_customer_count=len(regular), regular_definition=payload["regular_definition"],
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        caveats=["Dataset already explored; chronological isolation is computational, not a newly untouched confirmation set",
            "Masked basket completion conditions on items already purchased and cannot estimate incremental upsell",
            "No verified purchase sequence within orders", "Unit economics fixed from Jan-Mar; availability is a manual scenario input",
            "Aggregate regular-customer context is not individual personalization", "No layout or customer movement outcomes observed"])
    payload["evaluation"] = report
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    (ROOT / "models/basket_recommender.json").write_text(encoded, encoding="utf-8")
    (ROOT / "app/recommendation_data.js").write_text("window.STORIES_RECOMMENDATIONS="+encoded+";\n", encoding="utf-8")
    (out / "recommendation_results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{k:v for k,v in r.items() if k != "cohorts"} for r in test_summaries]).to_csv(out/"recommendation_metrics.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k != "cohorts"} for r in validation_results]).to_csv(out/"recommendation_validation.csv", index=False)
    test_frames[selected_name].to_csv(out/"recommendation_test_by_order.csv", index=False)
    # Training-only pair registry for display research; no pair mining from test data.
    g = final["global_stats"]
    pairs = []
    for i, a in enumerate(final["names"]):
        for j in range(i+1, len(final["names"])):
            count = g["pair"][i,j]
            if count:
                expected = g["single"][i]*g["single"][j]/g["n"]
                pairs.append(dict(item_a=a, item_b=final["names"][j], pair_orders=int(count),
                    anchor_a_orders=int(g["single"][i]), anchor_b_orders=int(g["single"][j]),
                    expected_under_independence=float(expected), lift=float(count/expected),
                    evidence_status="Training co-occurrence; no causal or multiple-testing-confirmed pair claim"))
    pd.DataFrame(pairs).sort_values("pair_orders", ascending=False).to_csv(out/"recommendation_pairs.csv", index=False)
    selected_result = test_summaries[chosen]
    lines = ["# Basket recommendation evaluation", "", f"Selected on March training validation: **{selected[0]}**, shrinkage {selected[1]}.",
        "", "Jan–Mar fitting; Apr–Jun frozen evaluation. Sales labels and economics are Stories-only.", "",
        "| Method | Strength | Hit@3 | MRR | Recovered-item margin@3/order |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['strength']:g} | {r['hit_at_3']:.2%} | {r['reciprocal_rank']:.4f} | ${r['recovered_margin_at_3_usd']:.4f} |" for r in test_summaries]
    lines += ["", f"Test: {selected_result['orders']:,} multi-product orders; {selected_result['masked_items']:,} masked items. Each order has equal weight.",
        "", "Recovered-item margin is an offline relevance proxy. It is not additional purchases, offer conversion, incremental profit or company savings.",
        "", "Seven-day paired block intervals and cohort results are in recommendation_results.json. The test was previously explored by the team; future experimental confirmation is required.",
        "", "Display separation and cross-selling require separate randomized pilots. A strong co-purchase can represent something customers already buy, creating no additional sale."]
    (out/"recommendation_evaluation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"selected":selected, "test":selected_result, "artifacts":"models/basket_recommender.json and app/recommendation_data.js"}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=str(ROOT/"data/stories_transactions.csv"))
    r = sub.add_parser("recommend")
    r.add_argument("--branch", required=True)
    r.add_argument("--item", action="append", default=[])
    r.add_argument("--unavailable", action="append", default=[])
    r.add_argument("--daypart", default="all", choices=DAYPARTS)
    r.add_argument("--cohort", default="all", choices=["all", "regular"])
    args = parser.parse_args()
    if args.command == "train":
        train(args.data)
    else:
        payload = json.loads((ROOT/"models/basket_recommender.json").read_text(encoding="utf-8"))
        print(json.dumps(recommend(payload, args.branch, args.item, args.daypart, args.cohort, args.unavailable), indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
