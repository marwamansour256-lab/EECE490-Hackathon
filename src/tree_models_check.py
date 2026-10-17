"""Compare a decision tree and a random forest with the Poisson preparation model.

Same Jan-Mar training, same branch x category x day target, settings chosen only on the last
21 training dates, April-June scored once. Each tree model is tried two ways:
  stand-alone  : Poisson-criterion tree/forest on branch, category, weekday and day number
                 (trees cannot extrapolate growth beyond the last training date)
  on_poisson   : the tree/forest learns a multiplier on the Poisson forecast
                 (target actual / Poisson forecast, weighted by the forecast), so growth comes
                 from the Poisson model and the trees only correct it
Writes reports/tree_models_check.json. Run: python src/tree_models_check.py
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from strict_model import daily_panel, features, fit_engine, metric, ALPHAS
from xgboost_check import tree_features

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {
    'decision_tree': dict(max_depth=[4, 8, 12, None], min_samples_leaf=[5, 20, 50]),
    'random_forest': dict(max_depth=[8, 12, None], min_samples_leaf=[5, 20], n_estimators=[300]),
}


def make(kind, params, multiplier):
    common = dict(random_state=490, criterion='squared_error' if multiplier else 'poisson')
    if kind == 'decision_tree':
        return DecisionTreeRegressor(**common, **params)
    return RandomForestRegressor(**common, n_jobs=4, max_features=0.8, **params)


def main():
    raw = pd.read_csv(ROOT/'data/stories_transactions.csv', parse_dates=['Date'])
    dates = pd.date_range(raw.Date.min(), raw.Date.max())
    cut = len(dates)//2
    train_dates, test_dates = dates[:cut], dates[cut:]
    training, testing = raw[raw.Date.isin(train_dates)], raw[raw.Date.isin(test_dates)]
    catalog = training[['Branch', 'Category', 'Product']].drop_duplicates()
    start = str(train_dates.min().date())
    branches, categories = sorted(catalog.Branch.unique()), sorted(catalog.Category.unique())
    outer_train = daily_panel(training, catalog, train_dates).reset_index(drop=True)
    outer_test = daily_panel(testing, catalog, test_dates).reset_index(drop=True)
    inner_cut = train_dates[-21]
    itr = outer_train[outer_train.Date.lt(inner_cut)].reset_index(drop=True)
    iva = outer_train[outer_train.Date.ge(inner_cut)].reset_index(drop=True)
    tf = lambda f: tree_features(f, start, branches, categories)

    pscores = []
    for a in ALPHAS:
        m = fit_engine(itr, a, start, categories)
        pscores.append((metric(iva.actual, m.predict(features(iva, start, categories)[0]))['poisson_deviance'], a))
    pa = min(pscores)[1]
    p_inner, p_outer = fit_engine(itr, pa, start, categories), fit_engine(outer_train, pa, start, categories)
    pin = lambda m, f: np.maximum(m.predict(features(f, start, categories)[0]), 1e-6)

    def fit_predict(kind, params, multiplier, train, test, pmodel):
        model = make(kind, params, multiplier)
        if multiplier:
            base_tr, base_te = pin(pmodel, train), pin(pmodel, test)
            model.fit(tf(train), train.actual/base_tr, sample_weight=base_tr)
            return np.maximum(model.predict(tf(test)), 0)*base_te
        model.fit(tf(train), train.actual)
        return np.maximum(model.predict(tf(test)), 1e-8)

    tuning, results = [], {}
    for kind, grid in GRIDS.items():
        for multiplier in [False, True]:
            name = kind + ('_on_poisson' if multiplier else '')
            best = None
            for values in itertools.product(*grid.values()):
                params = dict(zip(grid, values))
                r = metric(iva.actual, fit_predict(kind, params, multiplier, itr, iva, p_inner))
                tuning.append(dict(method=name, **{k: (v if v is not None else 'none') for k, v in params.items()}, **r))
                if best is None or r['poisson_deviance'] < best[0]:
                    best = (r['poisson_deviance'], params)
            pred = fit_predict(kind, best[1], multiplier, outer_train, outer_test, p_outer)
            results[name] = dict(selected={k: (v if v is not None else 'none') for k, v in best[1].items()},
                                 validation_deviance=best[0], test=metric(outer_test.actual, pred))

    ppred = pin(p_outer, outer_test)
    results['poisson (original)'] = dict(selected_alpha=pa, test=metric(outer_test.actual, ppred))
    calib = json.loads((ROOT/'reports/strict_half_results.json').read_text(encoding='utf-8'))['v2']['calibration_factor']
    results['poisson v2 (selected)'] = dict(selected_alpha=pa, calibration=calib, test=metric(outer_test.actual, ppred*calib))

    out = dict(question='Do a decision tree or a random forest beat the Poisson preparation model on the same frozen split?',
               level='branch x category x day', train=f'{start} to {train_dates.max().date()}',
               test=f'{test_dates.min().date()} to {test_dates.max().date()}',
               selection='settings chosen by Poisson deviance on the last 21 training dates; April-June scored once',
               results=results, tuning=tuning)
    (ROOT/'reports/tree_models_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    table = pd.DataFrame({k: v['test'] for k, v in results.items()}).T[['wape_pct', 'bias_pct', 'poisson_deviance']]
    print(table.round(3).to_string())


if __name__ == '__main__':
    main()
