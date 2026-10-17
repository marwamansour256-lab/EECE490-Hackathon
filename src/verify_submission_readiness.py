"""Inspect the submitted evidence with Python's standard library only.

This does not replace the numerical/model tests in verify_submission.py.
It permits inspection of saved evidence without installing model dependencies.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-6)


def audit():
    checks = []
    orders = {}
    train_lines = test_lines = 0
    raw_lines = 0
    for row in rows("data/stories_transactions.csv"):
        raw_lines += 1
        quantity = float(row["Quantity"])
        revenue = quantity * float(row["Unit Price (USD)"]) * (1 - float(row["Discount Rate"]))
        assert abs(revenue - float(row["Line Revenue (USD)"])) <= 0.011
        assert abs(revenue - quantity * float(row["Unit Cost (USD)"]) - float(row["Line Gross Profit (USD)"])) <= 0.011
        identity = (row["Date"], row["Branch"])
        order = orders.setdefault(row["Transaction ID"], [identity, set()])
        assert order[0] == identity, "Order crosses dates or branches"
        order[1].add(row["Product"])
        if row["Date"] <= "2026-03-31":
            train_lines += 1
        else:
            test_lines += 1
    assert (raw_lines, len(orders), train_lines, test_lines) == (51177, 33887, 22395, 28782)
    checks.append("51177 raw lines / 33887 complete orders; line economics and 22395/28782 chronological split reconcile")

    strict = read_json("reports/strict_half_results.json")
    assert strict["train"]["end"] == "2026-03-31" and strict["test"]["start"] == "2026-04-01"
    assert strict["selected_method_on_inner_validation"] == "regularized_poisson_v2"
    manifest = read_json("reports/strict_half_manifest.json")
    # Accept only Git line-ending conversion for text; binary bytes must match.
    for name, expected in manifest["sha256"].items():
        data = (ROOT / name).read_bytes()
        variants = [data]
        if Path(name).suffix in {".csv", ".py", ".json"}:
            lf = data.replace(b"\r\n", b"\n")
            variants.extend([lf, lf.replace(b"\n", b"\r\n")])
        assert expected in {hashlib.sha256(value).hexdigest() for value in variants}, name
    checks.append("Saved preparation artifact, model source, input and output hashes match the recorded numerical verification")

    seen = set()
    category_totals = defaultdict(float)
    category_means = {}
    finance = defaultdict(Counter)
    perishable = {"Bakery", "Desserts", "Sandwiches", "Breakfast"}
    for row in rows("reports/strict_half_predictions.csv"):
        key = tuple(row[k] for k in ("Date", "Branch", "Product", "method"))
        assert key not in seen and "2026-04-01" <= row["Date"] <= "2026-06-30"
        seen.add(key)
        assert close(row["actual"], float(row["sold_units"]) + float(row["unmet_units"]))
        assert close(row["prepare_units"], float(row["sold_units"]) + float(row["unsold_units"]))
        assert close(row["simulated_contribution"], float(row["price"]) * float(row["sold_units"]) - float(row["cost"]) * float(row["prepare_units"]))
        category_key = tuple(row[k] for k in ("Date", "Branch", "Category", "method"))
        category_totals[category_key] += float(row["predicted_mean_units"])
        category_means[category_key] = float(row["predicted_category_units"])
        if row["Category"] in perishable:
            for field in ("actual", "prepare_units", "sold_units", "unsold_units", "unmet_units", "simulated_contribution"):
                finance[row["method"]][field] += float(row[field])
    assert len(seen) == 49140
    assert all(close(value, category_means[key]) for key, value in category_totals.items())
    for row in rows("reports/strict_half_finance.csv"):
        for field, value in finance[row["method"]].items():
            assert close(value, row[field]), (row["method"], field)
        assert close(row["fill_rate_pct"], 100 * finance[row["method"]]["sold_units"] / finance[row["method"]]["actual"])
    checks.append("49140 prediction rows conserve demand, stock and contribution; category sums and four finance comparisons reconcile")

    recommendation = read_json("reports/recommendation_results.json")
    selected = next(r for r in recommendation["test"] if r["method"] == recommendation["selected_method"] and r["strength"] == recommendation["selected_strength"])
    evaluation = list(rows("reports/recommendation_test_by_order.csv"))
    expected_orders = sum(date > "2026-03-31" and len(products) > 1 for (date, branch), products in orders.values())
    assert len(evaluation) == selected["orders"] == expected_orders == 8801
    for metric in ("hit_at_3", "reciprocal_rank", "recovered_margin_at_3_usd"):
        assert close(sum(float(row[metric]) for row in evaluation) / len(evaluation), selected[metric])
    checks.append("8801 multi-product test orders and selected recommendation relevance/margin-proxy metrics reconcile")

    exports = [("app/recommendation_data.js", "STORIES_RECOMMENDATIONS", "models/basket_recommender.json"),
               ("app/wejha_data.js", "STORIES_WEJHA", "reports/wejha_layouts.json"),
               ("app/preparation_data.js", "STORIES_PREPARATION", "reports/growth_preparation_cards.json")]
    for script, variable, source in exports:
        text = (ROOT / script).read_text(encoding="utf-8")
        payload = re.split(r"window\." + variable + r"\s*=\s*", text, maxsplit=1)[1].strip().rstrip(";")
        assert json.loads(payload) == read_json(source), script
    checks.append("All three generated browser payloads equal their saved JSON artifacts")

    docs = ["README.md", "PROJECT_REPORT.md", "SUBMISSION.md", "SUBMISSION_AUDIT.md", "MODEL_CARD.md", "EVALUATION.md", "TEAM.md"]
    for name in docs:
        text = (ROOT / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^\s)]+)\)", text):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            assert (ROOT / Path(name).parent / target.split("#", 1)[0]).exists(), (name, target)
    checks.append("Submission entrypoint documents have no missing local file links")
    assert read_json("reports/basket_pilot_template_check.json")["incremental_profit_measured"] is False
    checks.append("Pilot template explicitly reports no observed incremental-profit result")
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "python": sys.version.split()[0], "status": "Saved-evidence checks passed",
            "checks_passed": checks,
            "scope": "Standard-library evidence audit; does not execute saved sklearn models or replace the 44-test numerical suite"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
