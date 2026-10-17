"""Fit and evaluate an intraday Gamma-Poisson model before profit-policy replay.

April-May chooses forecast family, shared dispersion and whether updating helps.
June is a chronological retrospective check, NOT a fresh holdout after prior work.
Recorded early sales are assumed uncensored for forecast scoring; inventory replay
instead feeds only fulfilled early sales and treats stock exhaustion as censoring.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from demand_model import panel, PERISHABLES
from bayesian_policy import optimize_adaptive, realize_adaptive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports'
SHAPES = [1., 2., 5., 10., 25., 100., 1000.]


def make_frame(raw):
    raw = raw.copy()
    raw['early_flag'] = raw.Time.str[:2].astype(int).lt(16)
    cat = panel(raw, ['Branch', 'Category'])
    early = raw[raw.early_flag].groupby(['Date', 'Branch', 'Category']).Quantity.sum().rename('early').reset_index()
    cat = cat.merge(early, on=['Date', 'Branch', 'Category'], how='left').fillna({'early': 0})
    cat = cat.sort_values(['Branch', 'Category', 'Date'])
    for source in ['early', 'actual']:
        cat[source+'_56'] = cat.groupby(['Branch', 'Category'])[source].transform(lambda s: s.shift(1).rolling(56, min_periods=1).sum())
    global_e = cat.groupby(['Date', 'Category']).early_56.transform('sum')
    global_y = cat.groupby(['Date', 'Category']).actual_56.transform('sum')
    cat['early_share'] = (cat.early_56+20*global_e/global_y)/(cat.actual_56+20)
    sku = panel(raw, ['Branch', 'Category', 'Product'])
    total = sku.groupby(['Date', 'Branch', 'Category']).history28.transform('sum')
    count = sku.groupby(['Date', 'Branch', 'Category']).Product.transform('size')
    sku['item_share'] = (sku.history28+1)/(total+count)
    early_sku = raw[raw.early_flag].groupby(['Date', 'Branch', 'Product']).Quantity.sum().rename('early').reset_index()
    sku = sku.merge(early_sku, on=['Date', 'Branch', 'Product'], how='left').fillna({'early': 0})
    sku['late'] = sku.actual-sku.early
    sku = sku.merge(cat[['Date', 'Branch', 'Category', 'early_share']], on=['Date', 'Branch', 'Category'], validate='many_to_one')
    forecast = pd.read_csv(OUT/'model_predictions.csv', parse_dates=['Date'])
    forecast = forecast[['Date', 'Branch', 'Category', 'internal', 'four_weekdays']].rename(columns={'four_weekdays': 'strong_baseline'})
    sku = sku.merge(forecast, on=['Date', 'Branch', 'Category'], how='inner', validate='many_to_one')
    for model in ['internal', 'strong_baseline']:
        sku[model] = np.maximum(.000001, sku[model]*sku.item_share)
    sku = sku[sku.Category.isin(PERISHABLES)].copy()
    history = raw[raw.Date.lt('2026-04-01')].copy()
    history['ingredient_cost'] = history.Quantity*history['Unit Cost (USD)']
    economics = history.groupby(['Branch', 'Product']).agg(units=('Quantity', 'sum'), revenue=('Line Revenue (USD)', 'sum'), total_cost=('ingredient_cost', 'sum'))
    economics['price'] = economics.revenue/economics.units
    economics['cost'] = economics.total_cost/economics.units
    return sku.merge(economics[['price', 'cost']], on=['Branch', 'Product'], validate='many_to_one').sort_values(['Date', 'Branch', 'Product']).reset_index(drop=True)


def predictive_arrays(frame, forecast, shape, adaptive):
    total = frame[forecast].to_numpy()
    early_mean = total*frame.early_share.to_numpy()
    late_mean = total-early_mean
    if shape is None:
        dist = poisson(late_mean)
        mean = late_mean
    else:
        n = shape+frame.early.to_numpy() if adaptive else np.full(len(frame), shape)
        rate = shape+early_mean if adaptive else np.full(len(frame), shape)
        mean = late_mean*n/rate
        dist = nbinom(n, rate/(rate+late_mean))
    y = frame.late.to_numpy()
    return mean, dist.logpmf(y), dist.ppf(.10), dist.ppf(.90)


def scores(frame, forecast, shape, adaptive):
    mean, logp, low, high = predictive_arrays(frame, forecast, shape, adaptive)
    y = frame.late.to_numpy()
    return dict(negative_log_likelihood=float(-logp.mean()), wape_pct=float(100*np.abs(y-mean).sum()/y.sum()),
                mae_units=float(np.abs(y-mean).mean()), coverage_p10_p90_pct=float(100*((y >= low)&(y <= high)).mean()),
                mean_interval_width=float((high-low).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    raw = pd.read_csv(ROOT/'data/stories_transactions.csv', parse_dates=['Date'])
    frame = make_frame(raw)
    frame.to_csv(OUT/'intraday_features.csv', index=False)
    validation = frame[frame.Date.lt('2026-06-01')]
    june = frame[frame.Date.ge('2026-06-01')]
    candidates = []
    for forecast in ['strong_baseline', 'internal']:
        for shape, adaptive in [(None, False)]+[(k, flag) for k in SHAPES for flag in (False, True)]:
            name = forecast+('_poisson' if shape is None else f'_k{shape:g}_'+('adaptive' if adaptive else 'static'))
            candidates.append(dict(name=name, forecast=forecast, shape=shape, adaptive=adaptive,
                                   validation=scores(validation, forecast, shape, adaptive),
                                   june=scores(june, forecast, shape, adaptive)))
    selected = min(candidates, key=lambda c: c['validation']['negative_log_likelihood'])
    best_adaptive = min([c for c in candidates if c['adaptive']], key=lambda c: c['validation']['negative_log_likelihood'])
    best_static = min([c for c in candidates if not c['adaptive']], key=lambda c: c['validation']['negative_log_likelihood'])
    selection = dict(selected=selected, best_adaptive=best_adaptive, best_static=best_static,
                     protocol='Select on April-May late-demand log score only; June is retrospective and already inspected in earlier work.',
                     candidate_count=len(candidates), candidates=candidates,
                     assumptions=['Gamma-Poisson shared daily intensity is a hypothesis, with a non-updating Poisson control.',
                                  'Forecast scoring treats recorded early and late sales as uncensored demand.',
                                  'Inventory replay updates on fulfilled early sales only; sold-out early observations are censored.',
                                  'Prior forecasts and intraday shares use only earlier dates. The early observation is allowed only at the 16:00 decision.',
                                  'The numerical model does not infer a news event caused a demand change.'],
                     technical_sources=['https://mc-stan.org/docs/2_36/stan-users-guide/posterior-prediction.html',
                                        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html'])
    (OUT/'intraday_model_results.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: selection[key] for key in ['selected', 'best_adaptive', 'best_static']}, indent=2), flush=True)
    rows = []
    for candidate in candidates:
        for period in ['validation', 'june']:
            rows.append(dict(name=candidate['name'], period=period, **candidate[period]))
    pd.DataFrame(rows).to_csv(OUT/'intraday_model_metrics.csv', index=False)
    if args.score_only:
        return
    # Preserve an ablation even if model selection rejects adaptive learning.
    compared = {c['name']: c for c in [selected, best_adaptive, best_static]}
    # These controls match stage marginals; shared-state dependence still differs.
    for target in [selected, best_adaptive]:
        if target['shape'] is not None:
            counterpart = next(c for c in candidates if c['forecast'] == target['forecast']
                               and c['shape'] == target['shape'] and c['adaptive'] != target['adaptive'])
            compared[counterpart['name']] = counterpart
    compared['strong_baseline_poisson'] = next(c for c in candidates if c['name'] == 'strong_baseline_poisson')
    frozen = dict(best_adaptive, name=best_adaptive['name']+'_frozen_signal', use_observation=False)
    compared[frozen['name']] = frozen
    replay, cards, predictions = [], [], []
    for name, candidate in compared.items():
        print('Replaying '+name, flush=True)
        means, logp, lows, highs = predictive_arrays(june, candidate['forecast'], candidate['shape'], candidate['adaptive'] and candidate.get('use_observation', True))
        for index, row in enumerate(june.itertuples()):
            policy = optimize_adaptive(getattr(row, candidate['forecast']), row.early_share, row.price, row.cost,
                                       candidate['shape'], candidate['adaptive'], use_observation=candidate.get('use_observation', True))
            result = realize_adaptive(policy, row.early, row.late)
            replay.append(dict(Date=str(row.Date.date()), Branch=row.Branch, Category=row.Category, Product=row.Product,
                               candidate=name, actual=row.actual, early_demand=row.early, late_demand=row.late, **result))
            predictions.append(dict(Date=str(row.Date.date()), Branch=row.Branch, Product=row.Product, candidate=name,
                                    late_actual=row.late, late_mean=means[index], p10=lows[index], p90=highs[index], log_score=logp[index]))
            if row.Date == pd.Timestamp('2026-06-30'):
                cards.append(dict(date='2026-06-30', branch=row.Branch, product=row.Product, category=row.Category,
                                  candidate=name, selected=name == selected['name'],
                                  status='Historical adaptive policy replay; not a production recommendation', **policy))
    replay = pd.DataFrame(replay)
    replay.to_csv(OUT/'intraday_policy_replay.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT/'intraday_predictions.csv', index=False)
    measures = ['actual', 'opening_units', 'replenished_units', 'sold_units', 'unsold_units', 'unmet_units', 'contribution', 'one_batch_contribution', 'incremental_contribution']
    summary = replay.groupby('candidate')[measures].sum().reset_index()
    summary['fill_rate_pct'] = 100*summary.sold_units/summary.actual
    summary.to_csv(OUT/'intraday_policy_summary.csv', index=False)
    contrasts = []
    indexed = summary.set_index('candidate')
    for shape in sorted({c['shape'] for c in compared.values() if c['shape'] is not None}):
        for forecast in ['strong_baseline', 'internal']:
            a = forecast+f'_k{shape:g}_adaptive'
            b = forecast+f'_k{shape:g}_static'
            if a in indexed.index and b in indexed.index:
                contrasts.append(dict(forecast=forecast, shape=shape,
                    adaptive_minus_static_contribution=float(indexed.loc[a, 'contribution']-indexed.loc[b, 'contribution']),
                    adaptive_fill_rate_pct=float(indexed.loc[a, 'fill_rate_pct']),
                    static_fill_rate_pct=float(indexed.loc[b, 'fill_rate_pct'])))
    selection['retrospective_policy_summary'] = summary.to_dict('records')
    selection['matched_stage_marginal_contrasts'] = contrasts
    selection['stage_marginal_contrast_warning'] = 'These match marginal stage dispersion but change early-late dependence. They are not pure value-of-information contrasts.'
    a = indexed.loc[best_adaptive['name']]
    b = indexed.loc[frozen['name']]
    selection['same_joint_prior_information_ablation'] = dict(
        adaptive_candidate=best_adaptive['name'], frozen_candidate=frozen['name'],
        adaptive_minus_frozen_contribution=float(a.contribution-b.contribution),
        adaptive_fill_rate_pct=float(a.fill_rate_pct), frozen_fill_rate_pct=float(b.fill_rate_pct),
        explanation='Same mean, shape, joint early-late prior and one-batch comparator. Frozen chooses topups from the marginal late distribution; adaptive uses the stockout-aware posterior. Stage service constraints use the information available to each policy.')
    selection['policy_costs'] = dict(setup_usd_per_sku=.25, extra_usd_per_topup_unit=.10, salvage_fraction=0., expected_fill_floor=.90)
    selection['policy_assumptions'] = ['Instant 16:00 replenishment; no total labor/capacity constraint; sales treated as demand.',
        'Stage expected-fill constraints versus daily expected-fill comparator are different timing constraints.',
        'Policy developed after examining June. No future ROI or independent confirmation claim.',
        'Selection used predictive log score, not retrospective June profit. Profit and realized fulfillment are reported jointly.']
    (OUT/'intraday_model_results.json').write_text(json.dumps(selection, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'intraday_decision_cards.json').write_text(json.dumps(cards, indent=2, allow_nan=False), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
