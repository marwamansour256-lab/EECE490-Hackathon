"""Compare linear regression with the Poisson preparation model under the same frozen split.

Same features as src/strict_model.py (branch/category, category x weekday, branch x weekday,
growth trends), same Jan-Mar training, same last-21-training-days selection, April-June scored
once. Two linear variants:
  linear_counts  : ridge regression on daily units (can predict negative values; clipped at 0)
  linear_log     : ridge regression on log(units + 1), back-transformed
Writes reports/linear_baseline_check.json. Run: python src/linear_baseline_check.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from strict_model import daily_panel, features, fit_engine, metric, CATEGORICAL, ALPHAS

ROOT = Path(__file__).resolve().parents[1]
RIDGE_ALPHAS = [0.01, 0.1, 1., 10., 100.]


def fit_linear(train, alpha, start, categories, log_target):
    x, numeric = features(train, start, categories)
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                             ('num', StandardScaler(), numeric)])
    model = make_pipeline(pre, Ridge(alpha=alpha))
    y = np.log1p(train.actual) if log_target else train.actual
    model.fit(x, y)
    return model


def predict_linear(model, frame, start, categories, log_target):
    raw = model.predict(features(frame, start, categories)[0])
    return np.maximum(np.expm1(raw) if log_target else raw, 1e-8)


def main():
    raw = pd.read_csv(ROOT/'data/stories_transactions.csv', parse_dates=['Date'])
    dates = pd.date_range(raw.Date.min(), raw.Date.max())
    cut = len(dates)//2
    train_dates, test_dates = dates[:cut], dates[cut:]
    training, testing = raw[raw.Date.isin(train_dates)], raw[raw.Date.isin(test_dates)]
    catalog = training[['Branch', 'Category', 'Product']].drop_duplicates()
    start, categories = str(train_dates.min().date()), sorted(catalog.Category.unique())
    outer_train = daily_panel(training, catalog, train_dates)
    outer_test = daily_panel(testing, catalog, test_dates)
    inner_cut = train_dates[-21]
    inner_train, inner_valid = outer_train[outer_train.Date.lt(inner_cut)], outer_train[outer_train.Date.ge(inner_cut)]

    results, tuning = {}, []
    for name, log_target in [('linear_counts', False), ('linear_log', True)]:
        scores = []
        for a in RIDGE_ALPHAS:
            m = fit_linear(inner_train, a, start, categories, log_target)
            r = metric(inner_valid.actual, predict_linear(m, inner_valid, start, categories, log_target))
            scores.append((r['poisson_deviance'], a))
            tuning.append(dict(method=name, alpha=a, **r))
        best = min(scores)[1]
        m = fit_linear(outer_train, best, start, categories, log_target)
        results[name] = dict(selected_alpha=best, test=metric(outer_test.actual, predict_linear(m, outer_test, start, categories, log_target)))

    # Poisson reference with the same selection rule (as in strict_model.py)
    pscores = []
    for a in ALPHAS:
        m = fit_engine(inner_train, a, start, categories)
        pscores.append((metric(inner_valid.actual, m.predict(features(inner_valid, start, categories)[0]))['poisson_deviance'], a))
    pa = min(pscores)[1]
    pm = fit_engine(outer_train, pa, start, categories)
    ppred = pm.predict(features(outer_test, start, categories)[0])
    results['poisson (original)'] = dict(selected_alpha=pa, test=metric(outer_test.actual, ppred))
    strict = json.loads((ROOT/'reports/strict_half_results.json').read_text(encoding='utf-8'))
    calib = strict['v2']['calibration_factor']
    results['poisson v2 (selected)'] = dict(selected_alpha=pa, calibration=calib, test=metric(outer_test.actual, ppred*calib))

    out = dict(question='Does linear regression beat the Poisson preparation model on the same frozen split?',
               level='branch x category x day', train=f'{start} to {train_dates.max().date()}',
               test=f'{test_dates.min().date()} to {test_dates.max().date()}',
               selection='alpha chosen by Poisson deviance on the last 21 training dates; April-June scored once',
               results=results, tuning=tuning)
    (ROOT/'reports/linear_baseline_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    table = pd.DataFrame({k: v['test'] for k, v in results.items()}).T[['wape_pct', 'bias_pct', 'poisson_deviance']]
    print(table.round(3).to_string())


if __name__ == '__main__':
    main()
