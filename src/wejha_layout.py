"""Compare Stories front-display hypotheses using training-only basket pairs.

Weighted pair distance is a geometric exposure proxy. No observed revenue,
walking path or causal uplift is inferred. The default demo uses bakery items
as ambient candidates, pending the company's food-storage confirmation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_layout(slots, products, fixed=None, zones=None, eligible_zones=None):
    if len(slots) != len(products) or set(slots) != set(products) or len(set(slots)) != len(slots):
        raise ValueError("Every eligible product must occur in exactly one slot")
    for slot, product in (fixed or {}).items():
        if slots[int(slot)] != product:
            raise ValueError("A fixed product has been moved")
    if zones:
        if len(zones) != len(slots):
            raise ValueError("Slot-zone count does not match slot count")
        for i, product in enumerate(slots):
            if zones[i] not in (eligible_zones or {}).get(product, []):
                raise ValueError(f"Product {product} incompatible with zone {zones[i]}")
    return True


def distance_score(slots, pairs):
    if len(slots) < 2:
        return 0.0
    positions = {p: i for i, p in enumerate(slots)}
    total = sum(p["weight"] for p in pairs if p["item_a"] in positions and p["item_b"] in positions)
    if total <= 0:
        return 0.0
    value = sum(p["weight"] * abs(positions[p["item_a"]]-positions[p["item_b"]]) / (len(slots)-1)
                for p in pairs if p["item_a"] in positions and p["item_b"] in positions)
    return float(value / total)


def optimize(products, pairs, maximize=True, fixed=None, zones=None, eligible_zones=None):
    """Exact enumeration up to eight products; constrained swap search above that."""
    products = list(products)
    fixed = {int(k): v for k, v in (fixed or {}).items()}
    if any(i < 0 or i >= len(products) for i in fixed):
        raise ValueError("Fixed slot is outside display")
    if len(set(fixed.values())) != len(fixed) or any(p not in products for p in fixed.values()):
        raise ValueError("Fixed products must be unique eligible products")
    sign = 1 if maximize else -1
    best, best_score = None, -np.inf

    def consider(candidate):
        nonlocal best, best_score
        try:
            validate_layout(candidate, products, fixed, zones, eligible_zones)
        except ValueError:
            return
        score = sign * distance_score(candidate, pairs)
        if score > best_score + 1e-12:
            best, best_score = list(candidate), score

    if len(products) <= 8:
        free_slots = [i for i in range(len(products)) if i not in fixed]
        free_products = sorted(set(products) - set(fixed.values()))
        for perm in itertools.permutations(free_products):
            candidate = [None] * len(products)
            for i, p in fixed.items():
                candidate[i] = p
            for i, p in zip(free_slots, perm):
                candidate[i] = p
            consider(candidate)
        method = "Exact enumeration within supplied slots and constraints"
    else:
        rng = np.random.default_rng(490)
        for _ in range(20):
            candidate = [None] * len(products)
            free = sorted(set(products)-set(fixed.values()))
            rng.shuffle(free)
            for i, p in fixed.items():
                candidate[i] = p
            for i in range(len(candidate)):
                if candidate[i] is None:
                    candidate[i] = free.pop()
            consider(candidate)
        if best is not None:
            improved = True
            while improved:
                previous = best_score
                for i, j in itertools.combinations(range(len(products)), 2):
                    if i in fixed or j in fixed:
                        continue
                    candidate = list(best)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    consider(candidate)
                improved = best_score > previous + 1e-12
        method = "Seeded multi-start constrained swap search; not guaranteed global optimum"
    if best is None:
        raise ValueError("No feasible layout under supplied constraints")
    return best, method


def build(products=None, fixed=None, zones=None, eligible_zones=None):
    payload = json.loads((ROOT/"models/basket_recommender.json").read_text(encoding="utf-8"))
    categories = {p["product"]: p["category"] for p in payload["products"]}
    products = sorted(products or [p for p, c in categories.items() if c == "Bakery"])
    if len(products) < 2 or len(set(products)) != len(products):
        raise ValueError("Supply at least two distinct display products")
    if any(p not in categories for p in products):
        raise ValueError("Unknown display product")
    registry = pd.read_csv(ROOT/"reports/recommendation_pairs.csv")
    selected = registry[registry.item_a.isin(products) & registry.item_b.isin(products)].copy()
    # Shared demand volume is deliberate: high-frequency pairs are exposure anchors,
    # not a claim that a pair beats an independence baseline or implies extra sales.
    selected["weight"] = selected.pair_orders.astype(float)
    pairs = selected.to_dict("records")
    if not pairs:
        raise ValueError("No observed training pairs among display products")
    constraints = dict(fixed=fixed, zones=zones, eligible_zones=eligible_zones)
    adjacent, min_method = optimize(products, pairs, maximize=False, **constraints)
    separated, max_method = optimize(products, pairs, maximize=True, **constraints)
    rng = np.random.default_rng(490)
    random_scores, random_slots = [], []
    for _ in range(1000):
        candidate = list(products)
        rng.shuffle(candidate)
        try:
            validate_layout(candidate, products, **constraints)
        except ValueError:
            continue
        random_scores.append(distance_score(candidate, pairs))
        random_slots.append(candidate)
    # Strong constraints can make rejection sampling inefficient; a feasible layout
    # remains valid but is not falsely reported as a random sample.
    if random_scores:
        order = np.argsort(random_scores)
        representative = random_slots[int(order[len(order)//2])]
        random_name = "random_reference"
    else:
        representative = adjacent
        random_name = "feasible_reference_no_random_draw"
    descriptions = {
        "adjacent_pairs": "Clusters frequent co-purchases: lower geometric distance; may reduce search or staff effort.",
        "separated_pairs": "Spreads frequent co-purchases across the display: higher hypothetical exposure, possibly more friction.",
        random_name: "Seeded feasible arrangement; not Stories' current layout and not a sales baseline.",
    }
    layouts = [dict(name=name, slots=slots, weighted_pair_distance=distance_score(slots, pairs), interpretation=descriptions[name])
               for name, slots in [("adjacent_pairs", adjacent), ("separated_pairs", separated), (random_name, representative)]]
    result = dict(schema_version=1, status="Layout hypothesis; incremental sales and customer paths unmeasured",
        trained_through=payload["trained_through"], products=products, pairs=pairs, layouts=layouts,
        optimization=dict(adjacent=min_method, separated=max_method),
        score_formula="sum(pair_orders * abs(slot_a-slot_b)/(slots-1)) / sum(pair_orders)",
        random_reference=dict(draws=len(random_scores), mean=float(np.mean(random_scores)) if random_scores else None,
            p05=float(np.quantile(random_scores,.05)) if random_scores else None,
            p95=float(np.quantile(random_scores,.95)) if random_scores else None),
        constraints=dict(fixed=fixed or {}, zones=zones, eligible_zones=eligible_zones),
        assumptions=["Equal-width one-dimensional display slots; actual geometry unknown",
            "Default Bakery category is an ambient-display candidate set, pending food-safety confirmation",
            "No drinks or chilled desserts placed into ambient pastry slots by default",
            "Weights are training pair counts, not statistically confirmed affinity or incremental purchase probabilities",
            "Staff-served counters may create visual exposure without any customer walking",
            "No verified current arrangement, availability, refrigeration or allergen-separation constraints supplied"],
        pilot=dict(control="Company-supplied current layout", treatment="Separated candidate after physical feasibility check",
            primary_metric="Incremental contribution per eligible order, after extra labor, discounts and waste",
            guardrails=["queue/service time", "availability", "waste", "customer experience", "safe storage/accessibility"],
            inference="Randomize branch-day blocks balanced by branch/weekday; separate layout from recommendation experiments"))
    encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    (ROOT/"reports/wejha_layouts.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT/"app/wejha_data.js").write_text("window.STORIES_WEJHA="+encoded+";\n", encoding="utf-8")
    (ROOT/"reports/wejha_layouts.md").write_text("# Display-layout experiment\n\n" + result["status"] + "\n\n" +
        "\n\n".join(f"**{l['name']}**: {l['weighted_pair_distance']:.4f} normalized weighted distance.\n\n" + " → ".join(l['slots']) for l in layouts) +
        "\n\nHigher distance is the proposed exposure mechanism, not a demonstrated profit objective. Pair weights measure existing co-purchases. Actual customer paths, current layout and offer acceptance are absent.\n", encoding="utf-8")
    print(json.dumps({"products":len(products), "pairs":len(pairs), "layouts":layouts}, indent=2))
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", action="append", help="Repeat for a custom candidate display set")
    parser.add_argument("--constraints", help="JSON containing optional fixed, zones and eligible_zones")
    args = parser.parse_args()
    constraints = json.loads(Path(args.constraints).read_text(encoding="utf-8")) if args.constraints else {}
    build(args.product, **constraints)
