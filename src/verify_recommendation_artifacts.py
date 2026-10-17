"""Reconcile portable recommendation, layout, prep and integration artifacts."""
import importlib.metadata
import json
import math
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
import pandas as pd

from basket_recommender import recommend,load_orders,regular_ids
from provenance import matches_recorded_sha256,portable_sha256
from wejha_layout import validate_layout,distance_score

ROOT=Path(__file__).resolve().parents[1]


def script_payload(path,name):
    text=path.read_text(encoding="utf-8").strip()
    prefix="window."+name+"="
    assert text.startswith(prefix) and text.endswith(";"),str(path)
    return json.loads(text[len(prefix):-1])


def main():
    model=json.loads((ROOT/"models/basket_recommender.json").read_text(encoding="utf-8"))
    assert model==script_payload(ROOT/"app/recommendation_data.js","STORIES_RECOMMENDATIONS")
    result=json.loads((ROOT/"reports/recommendation_results.json").read_text(encoding="utf-8"))
    assert model["evaluation"]==result
    assert matches_recorded_sha256(ROOT/"data/stories_transactions.csv",result["source_sha256"])
    assert model["trained_through"]=="2026-03-31" and result["test_dates"]==["2026-04-01","2026-06-30"]
    assert result["validation_dates"]==["2026-03-11","2026-03-31"]
    winner=max(result["validation"],key=lambda r:r["recovered_margin_at_3_usd"])
    assert winner["method"]==model["selected_method"] and winner["strength"]==model["shrinkage_strength"]
    catalog={p["product"] for p in model["products"]}
    assert len(catalog)==45 and len(model["contexts"])==24
    parity=[]
    for key,ctx in model["contexts"].items():
        assert set(ctx["by_anchor"])==catalog
        assert {r["product"] for r in ctx["baseline"]}==catalog
        for anchor,rows in ctx["by_anchor"].items():
            assert {r["product"] for r in rows}==catalog-{anchor}
            assert rows==sorted(rows,key=lambda r:(-r["score"],r["product"]))
            for r in rows:
                assert 0<=r["conditional_probability"]<=1 and 0<=r["baseline_probability"]<=1
                assert r["pair_orders"]<=r["anchor_orders"]
                assert math.isfinite(r["score"])
                if model["selected_method"]!="popularity":
                    assert abs(r["conditional_probability"]*max(0,r["unit_margin_usd"])-r["score"])<1e-6
        branch,part,cohort=key.split("|")
        for basket in [[],["Espresso"],["Butter Croissant","Chocolate Cookie"]]:
            unavailable=["Iced Matcha"]
            output=recommend(model,branch,basket,part,cohort,unavailable)
            assert all(r["product"] not in set(basket+unavailable) for r in output["recommendations"])
            parity.append(dict(branch=branch,daypart=part,cohort=cohort,basket=basket,unavailable=unavailable,
                expected=[r["product"] for r in output["recommendations"]]))
    assert '"customer"' not in json.dumps(model) and '"regular_ids"' not in json.dumps(model)
    raw,orders=load_orders(ROOT/"data/stories_transactions.csv")
    train=orders[orders.date.le("2026-03-31")]
    test=orders[orders.date.gt("2026-03-31")]
    assert len(regular_ids(train))==result["regular_customer_count"]
    frame=pd.read_csv(ROOT/"reports/recommendation_test_by_order.csv")
    assert len(frame)==test.products.map(len).gt(1).sum()
    summary=next(r for r in result["test"] if r["method"]==model["selected_method"] and r["strength"]==model["shrinkage_strength"])
    for metric in ["hit_at_3","reciprocal_rank","recovered_margin_at_3_usd"]:
        assert np.isclose(frame[metric].mean(),summary[metric])
    layout=json.loads((ROOT/"reports/wejha_layouts.json").read_text(encoding="utf-8"))
    assert layout==script_payload(ROOT/"app/wejha_data.js","STORIES_WEJHA")
    for item in layout["layouts"]:
        validate_layout(item["slots"],layout["products"],**layout["constraints"])
        assert np.isclose(item["weighted_pair_distance"],distance_score(item["slots"],layout["pairs"]))
    prep=json.loads((ROOT/"reports/growth_preparation_cards.json").read_text(encoding="utf-8"))
    assert prep==script_payload(ROOT/"app/preparation_data.js","STORIES_PREPARATION") and len(prep["cards"])==57
    for card in prep["cards"].values():
        assert card["trained_through"]=="2026-03-31"
        assert card["test_sales_used_in_forecast"] is False and card["recommendation_uplift_added_to_forecast"] is False
        assert len(card["at16_topup_by_remaining_stock"])==card["opening_prepare_units"]+1
    demo=json.loads((ROOT/"reports/growth_demo.json").read_text(encoding="utf-8"))
    assert demo["profit_scenario_for_top_offer"]["incremental_contribution_scenario_usd"]==0
    assert {r["product"] for r in demo["offers"]["recommendations"]}=={r["product"] for r in demo["preparation"]}
    (ROOT/"reports/recommendation_frontend_parity_cases.json").write_text(json.dumps(parity,indent=2),encoding="utf-8")
    paths=["src/basket_recommender.py","src/wejha_layout.py","src/growth_plan.py","data/stories_transactions.csv",
        "models/basket_recommender.json","app/recommendation_data.js","app/wejha_data.js","app/preparation_data.js",
        "reports/recommendation_results.json","reports/wejha_layouts.json","reports/growth_demo.json"]
    checks=["Training-only selection and frozen Jan-Mar artifact metadata",
        "All24 contexts/45 products satisfy ranking, support, probability and exclusion constraints",
        "Browser payload equals saved model; no customer identifiers exported",
        "8801 held-out multi-product order metrics and training-defined cohort reconcile",
        "Display constraints, distances and browser payload reconcile",
        "57 preparation cards reconcile and no assumed recommendation uplift enters demand",
        "Integrated offer/prep products agree; default incremental-profit scenario is zero"]
    manifest=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        packages={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","joblib"]},
        hash_convention="SHA256 of LF-normalized text; binary bytes unchanged",
        sha256={p:portable_sha256(ROOT/p) for p in paths},checks_passed=checks)
    (ROOT/"reports/recommendation_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
