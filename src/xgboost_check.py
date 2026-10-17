"""Compare XGBoost with the Poisson preparation model under the same frozen split.

Same Jan-Mar training, same branch x category x day target, settings chosen only on the last
21 training dates, April-June scored once. Variants:
  xgb_poisson          : gradient-boosted trees, Poisson objective, features = branch, category,
                         weekday, day number (trees cannot extrapolate growth beyond training dates)
  xgb_on_poisson       : boosted trees that correct the Poisson model (its log forecast is the
                         starting point / base margin); tests whether trees add anything on top
Writes reports/xgboost_check.json. Run: python src/xgboost_check.py
Requires: xgboost
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from strict_model import daily_panel, features, fit_engine, metric, ALPHAS

ROOT = Path(__file__).resolve().parents[1]
GRID = dict(max_depth=[2, 4, 6], learning_rate=[0.03, 0.1], n_estimators=[200, 600])


def tree_features(frame, start, branches, categories):
    x = pd.DataFrame(index=frame.index)
    for b in branches:
        x['b_'+b] = frame.Branch.eq(b).astype(float)
    for c in categories:
        x['c_'+c] = frame.Category.eq(c).astype(float)
    x['weekday'] = frame.Date.dt.dayofweek.astype(float)
    x['day_number'] = (frame.Date-pd.Timestamp(start)).dt.days.astype(float)
    return x


def fit_xgb(x, y, params, base_margin=None):
    model = xgb.XGBRegressor(objective='count:poisson', tree_method='hist', random_state=490, n_jobs=4,
                             subsample=0.9, colsample_bytree=0.9, **params)
    model.fit(x, y, base_margin=base_margin)
    return model


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

    # Poisson model, same selection rule as strict_model.py
    pscores = []
    for a in ALPHAS:
        m = fit_engine(itr, a, start, categories)
        pscores.append((metric(iva.actual, m.predict(features(iva, start, categories)[0]))['poisson_deviance'], a))
    pa = min(pscores)[1]
    p_inner = fit_engine(itr, pa, start, categories)
    p_outer = fit_engine(outer_train, pa, start, categories)

    def pois_log(model, frame):
        return np.log(np.maximum(model.predict(features(frame, start, categories)[0]), 1e-8))

    tuning, results = [], {}
    for name in ['xgb_poisson', 'xgb_on_poisson']:
        best = None
        for depth, lr, n in itertools.product(*GRID.values()):
            params = dict(max_depth=depth, learning_rate=lr, n_estimators=n)
            margin_tr = pois_log(p_inner, itr) if name == 'xgb_on_poisson' else None
            margin_va = pois_log(p_inner, iva) if name == 'xgb_on_poisson' else None
            m = fit_xgb(tree_features(itr, start, branches, categories), itr.actual, params, margin_tr)
            pred = m.predict(tree_features(iva, start, branches, categories), base_margin=margin_va)
            r = metric(iva.actual, pred)
            tuning.append(dict(method=name, **params, **r))
            if best is None or r['poisson_deviance'] < best[0]:
                best = (r['poisson_deviance'], params)
        params = best[1]
        margin_tr = pois_log(p_outer, outer_train) if name == 'xgb_on_poisson' else None
        margin_te = pois_log(p_outer, outer_test) if name == 'xgb_on_poisson' else None
        m = fit_xgb(tree_features(outer_train, start, branches, categories), outer_train.actual, params, margin_tr)
        pred = m.predict(tree_features(outer_test, start, branches, categories), base_margin=margin_te)
        results[name] = dict(selected=params, validation_deviance=best[0], test=metric(outer_test.actual, pred))

    ppred = p_outer.predict(features(outer_test, start, categories)[0])
    results['poisson (original)'] = dict(selected_alpha=pa, test=metric(outer_test.actual, ppred))
    calib = json.loads((ROOT/'reports/strict_half_results.json').read_text(encoding='utf-8'))['v2']['calibration_factor']
    results['poisson v2 (selected)'] = dict(selected_alpha=pa, calibration=calib, test=metric(outer_test.actual, ppred*calib))

    out = dict(question='Does XGBoost beat the Poisson preparation model on the same frozen split?',
               level='branch x category x day', train=f'{start} to {train_dates.max().date()}',
               test=f'{test_dates.min().date()} to {test_dates.max().date()}',
               selection='settings chosen by Poisson deviance on the last 21 training dates; April-June scored once',
               xgboost_version=xgb.__version__, results=results, tuning=tuning)
    (ROOT/'reports/xgboost_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    table = pd.DataFrame({k: v['test'] for k, v in results.items()}).T[['wape_pct', 'bias_pct', 'poisson_deviance']]
    print(table.round(3).to_string())
    for k in ['xgb_poisson', 'xgb_on_poisson']:
        print(k, 'selected on validation:', results[k]['selected'])


if __name__ == '__main__':
    main()
