"""Explain a historical probabilistic inventory decision and its intraday posterior."""
import argparse
import json
from pathlib import Path
import numpy as np
from bayesian_policy import optimize_adaptive, joint_demand, conditional_late

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', default='Harbour')
    p.add_argument('--product', default='Turkey & Cheese Sandwich')
    p.add_argument('--date', default='2026-06-30')
    p.add_argument('--candidate', choices=['selected', 'best-adaptive', 'seasonal-control'], default='selected')
    p.add_argument('--remaining', type=int)
    p.add_argument('--setup', type=float, default=.25)
    p.add_argument('--premium', type=float, default=.10)
    p.add_argument('--fill-floor', type=float, default=.90)
    args = p.parse_args()
    results = json.loads((ROOT/'reports/intraday_model_results.json').read_text(encoding='utf-8'))
    name = results['selected']['name'] if args.candidate == 'selected' else results['best_adaptive']['name'] if args.candidate == 'best-adaptive' else 'strong_baseline_poisson'
    cards_path = ROOT/'reports/intraday_decision_cards.json'
    if not cards_path.exists():
        p.error('Run python src/intraday_model.py to generate the policy cards first')
    cards = json.loads(cards_path.read_text(encoding='utf-8'))
    matches = [c for c in cards if (c['date'], c['branch'], c['product'], c['candidate']) == (args.date, args.branch, args.product, name)]
    if not matches:
        p.error('No historical card for that date, branch, product and candidate')
    card = matches[0]
    try:
        policy = optimize_adaptive(card['mean'], card['mean_early']/card['mean'] if card['mean'] else .5,
                                   card['price'], card['cost'], card['shape'], card['adaptive'],
                                   setup=args.setup, premium=args.premium, fill_floor=args.fill_floor)
    except ValueError as exc:
        p.error(str(exc))
    opening = policy['opening_units']
    output = dict(status=card['status'], date=args.date, branch=args.branch, product=args.product,
                  candidate=name, selected_on='April-May late-demand predictive log score',
                  adaptive_signal_enabled=policy['adaptive'], opening_prepare_units=opening,
                  prior_late_mean_units=policy['mean_late'],
                  at16_topup_by_remaining_stock=policy['replenish_by_remaining_stock'],
                  expected_two_stage_contribution=policy['expected_two_stage_contribution'],
                  expected_one_batch_contribution=policy['expected_one_batch_contribution'],
                  assumptions=dict(setup_usd_per_sku=args.setup, extra_usd_per_topup_unit=args.premium,
                                   expected_fill_floor=args.fill_floor, immediate_replenishment=True,
                                   capacity_unconstrained=True, actual_stockout_logs_unavailable=True),
                  comparison_status='Retrospective June evaluation; adaptive updating did not win the April-May model selection')
    if args.remaining is not None:
        if not 0 <= args.remaining <= opening:
            p.error(f'--remaining must be between 0 and opening stock {opening}')
        observed = opening-args.remaining
        sold_out = args.remaining == 0
        joint, _ = joint_demand(policy['mean_early'], policy['mean_late'], policy['shape'], policy['adaptive'])
        try:
            pmf = conditional_late(joint, observed, sold_out=sold_out)
        except ValueError as exc:
            p.error(str(exc))
        cumulative = np.cumsum(pmf)
        output.update(observed_early_sales_units=observed, early_stock_exhausted=sold_out,
                      observation_interpretation=f'Early demand >= {observed}' if sold_out else f'Early demand = {observed}, assuming no prior stockout or stock changes',
                      posterior_late_mean_units=float(np.arange(len(pmf))@pmf),
                      posterior_p10_units=int(np.searchsorted(cumulative, .10)),
                      posterior_p90_units=int(np.searchsorted(cumulative, .90)),
                      prepare_at16_units=policy['replenish_by_remaining_stock'][str(args.remaining)])
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
