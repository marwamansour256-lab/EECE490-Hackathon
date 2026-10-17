"""Use the trained half/half artifact without reading sales or external files."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from strict_model import features, baseline_predict
from profit_policy import optimize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', default='2026-06-30')
    p.add_argument('--branch', default='Harbour')
    p.add_argument('--product', default='Turkey & Cheese Sandwich')
    p.add_argument('--method', choices=['selected', 'regularized_poisson', 'seasonal_frozen', 'seasonal_growth'], default='selected')
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
    if method == 'regularized_poisson':
        category_mean = float(artifact['model'].predict(features(frame, artifact['start'], artifact['categories'])[0])[0])
    else:
        category_mean = float(baseline_predict(frame, artifact['baseline'], artifact['start'], method == 'seasonal_growth')[0])
    mean = category_mean*float(row.item_share)
    try:
        policy = optimize(mean, float(row.early_share), float(row.price), float(row.cost),
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
                  assumptions=dict(setup_usd_per_sku=args.setup, extra_usd_per_topup_unit=args.premium,
                                   expected_fill_floor=args.fill_floor, instant_replenishment=True, capacity_unconstrained=True))
    if args.remaining is not None:
        if not 0 <= args.remaining <= opening:
            p.error(f'--remaining must be between 0 and opening stock {opening}')
        output['prepare_at16_units'] = int(policy['replenish_by_remaining_stock'][args.remaining])
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
