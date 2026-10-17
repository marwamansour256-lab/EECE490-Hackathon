"""Use the trained half/half artifact without reading sales or external files."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from strict_model import category_mean, share_column, V2_METHOD
from profit_policy import optimize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', default='2026-06-30')
    p.add_argument('--branch', default='Harbour')
    p.add_argument('--product', default='Turkey & Cheese Sandwich')
    p.add_argument('--method', choices=['selected', 'regularized_poisson', V2_METHOD, 'seasonal_frozen', 'seasonal_growth'], default='selected')
    # Recommendation scenarios (ANALYSIS.md): assumptions, not measured effects. Demand is held constant.
    p.add_argument('--cost-multiplier', type=float, default=1.0, help='e.g. 1.10 if real ingredient costs rose 10 percent (ANALYSIS.md item 1)')
    p.add_argument('--price-multiplier', type=float, default=1.0, help='e.g. 1.05 for the proposed +5 percent price test (item 3)')
    p.add_argument('--no-discount', action='store_true', help='use the list price instead of the training average paid price (item 4)')
    p.add_argument('--remaining', type=int)
    p.add_argument('--setup', type=float, default=.25)
    p.add_argument('--premium', type=float, default=.10)
    p.add_argument('--fill-floor', type=float, default=.90)
    args = p.parse_args()
    path = ROOT/'models/strict_half_model.joblib'
    if not path.exists():
        p.error('Train the artifact first: python src/strict_model.py')
    artifact = joblib.load(path)
    try:
        date = pd.Timestamp(args.date)
    except ValueError:
        p.error('Invalid date')
    if date <= pd.Timestamp(artifact['train_end']):
        p.error('Choose a date after the March 31 training cutoff')
    params = artifact['sku_parameters']
    matches = params[params.Branch.eq(args.branch)&params.Product.eq(args.product)]
    if len(matches) != 1:
        p.error('Unknown branch/product; a cold-start rule is required')
    row = matches.iloc[0]
    frame = pd.DataFrame([dict(Date=date, Branch=args.branch, Category=row.Category)])
    method = artifact['selected_method'] if args.method == 'selected' else args.method
    if method == V2_METHOD and 'v2_calibration' not in artifact:
        p.error('Retrain the artifact (python src/strict_model.py) to use the v2 method')
    mean = float(category_mean(artifact, frame, method)[0])*float(row[share_column(method)])
    base_price = float(row['list_price']) if args.no_discount and 'list_price' in row.index else float(row.price)
    price = base_price*args.price_multiplier
    cost = float(row.cost)*args.cost_multiplier
    try:
        policy = optimize(mean, float(row.early_share), price, cost,
                          setup=args.setup, premium=args.premium, fill_floor=args.fill_floor)
    except ValueError as exc:
        p.error(str(exc))
    opening = policy['opening_units']
    output = dict(status='Scenario from frozen Jan-March model; operational feasibility and future validation required',
                  date=args.date, branch=args.branch, product=args.product, category=row.Category,
                  method=method, trained_through=artifact['train_end'], external_model_inputs=False,
                  test_sales_used_in_forecast=False, forecast_mean_units=mean,
                  opening_prepare_units=opening, one_batch_comparator_units=policy['one_batch_units'],
                  at16_topup_by_remaining_stock={str(i): int(q) for i, q in enumerate(policy['replenish_by_remaining_stock']) if i <= opening},
                  expected_two_stage_contribution=policy['expected_two_stage_contribution'],
                  expected_one_batch_contribution=policy['expected_one_batch_contribution'],
                  scenario=dict(cost_multiplier=args.cost_multiplier, price_multiplier=args.price_multiplier,
                                no_discount=args.no_discount, price_used=price, cost_used=cost,
                                note='Recommendation scenarios hold demand constant; customer response to price is not estimated.'),
                  assumptions=dict(setup_usd_per_sku=args.setup, extra_usd_per_topup_unit=args.premium,
                                   expected_fill_floor=args.fill_floor, instant_replenishment=True, capacity_unconstrained=True))
    if args.remaining is not None:
        if not 0 <= args.remaining <= opening:
            p.error(f'--remaining must be between 0 and opening stock {opening}')
        output['prepare_at16_units'] = int(policy['replenish_by_remaining_stock'][args.remaining])
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
