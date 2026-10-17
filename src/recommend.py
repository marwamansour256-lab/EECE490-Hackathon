"""Review a historical decision, or price a preparation scenario with supplied inputs.

This CLI never describes June decisions as today's production recommendations.
"""
import argparse
import json
from pathlib import Path
from profit_policy import optimize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', default='2026-06-30')
    p.add_argument('--branch', default='Harbour')
    p.add_argument('--product', default='Turkey & Cheese Sandwich')
    p.add_argument('--forecast', choices=['strong_baseline', 'internal'], default='strong_baseline')
    p.add_argument('--remaining', type=int, help='Observed inventory at 16:00, before replenishment')
    p.add_argument('--setup', type=float, default=.25, help='USD per SKU second batch')
    p.add_argument('--premium', type=float, default=.10, help='Extra USD per second-batch unit')
    p.add_argument('--salvage', type=float, default=0., help='Fraction of ingredient cost recovered at close')
    p.add_argument('--fill-floor', type=float, default=.90, help='Expected unit fill fraction, not probability of no stockout')
    p.add_argument('--mean', type=float, help='Supply with early-share, price and cost for a custom scenario')
    p.add_argument('--early-share', type=float)
    p.add_argument('--price', type=float)
    p.add_argument('--cost', type=float)
    args = p.parse_args()
    inputs = [args.mean, args.early_share, args.price, args.cost]
    if any(v is not None for v in inputs):
        if not all(v is not None for v in inputs):
            p.error('Custom scenarios require --mean, --early-share, --price and --cost together')
        mean, share, price, cost = inputs
        status = 'Custom scenario using user-supplied inputs; not a validated demand forecast'
    else:
        cards = json.loads((ROOT/'reports/decision_cards.json').read_text(encoding='utf-8'))
        matches = [c for c in cards if c['date'] == args.date and c['branch'] == args.branch
                   and c['product'] == args.product and c['forecast'] == args.forecast]
        if not matches:
            available = sorted({(c['date'], c['branch'], c['product']) for c in cards})
            p.error('No historical card matches. Available date/branch/product combinations: '+repr(available))
        card = matches[0]
        mean, price, cost = card['forecast_mean'], card['price'], card['cost']
        share = card['forecast_before16']/mean if mean else .5
        status = card['status']
    try:
        policy = optimize(mean, share, price, cost, args.setup, args.premium, args.salvage, args.fill_floor)
    except ValueError as exc:
        p.error(str(exc))
    opening = policy['opening_units']
    if args.remaining is not None and not 0 <= args.remaining <= opening:
        p.error(f'--remaining must be between 0 and the planned opening stock {opening}')
    table = {str(i): int(q) for i, q in enumerate(policy['replenish_by_remaining_stock']) if i <= opening}
    result = dict(status=status, date=args.date, branch=args.branch, product=args.product,
                  forecast=args.forecast if args.mean is None else 'user_supplied',
                  forecast_mean_units=mean, opening_prepare_units=opening,
                  one_batch_comparator_units=policy['one_batch_units'],
                  expected_two_stage_contribution=policy['expected_two_stage_contribution'],
                  expected_one_batch_contribution=policy['expected_one_batch_contribution'],
                  expected_incremental_contribution=policy['expected_two_stage_contribution']-policy['expected_one_batch_contribution'],
                  comparison_note='Same forecast and economics; two-stage service floor applies per stage, one-batch floor per day. Expected contribution is model-based, not guaranteed profit.',
                  at16_topup_by_remaining_stock=table,
                  assumptions=dict(setup_usd_per_sku=args.setup, extra_usd_per_topup_unit=args.premium,
                                   salvage_fraction_of_cost=args.salvage, expected_fill_floor=args.fill_floor,
                                   immediate_second_batch=True, capacity_unconstrained=True),
                  event_adjustment='None: event occurrence alone does not establish branch exposure or causal uplift')
    if args.remaining is not None:
        result['observed_remaining_units'] = args.remaining
        result['prepare_at16_units'] = table[str(args.remaining)]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
