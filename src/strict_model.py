"""Frozen chronological half/half experiment using ONE transaction CSV only.

First 90 of 181 dates train; last 91 dates test. The final 21 training dates
select hyperparameters. Refit once on all 90 training dates, then freeze every
parameter, item share, price/cost and intraday split. No test lags or external
files enter predictions; the whole test period is forecast at the cutoff.
"""
from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from demand_model import PERISHABLES
from profit_policy import fill_floor_stock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports'
ALPHAS = [0., .001, .01, .1, 1.]
CATEGORICAL = ['series', 'category_weekday', 'branch_weekday']
# v2 (teammate 1): item shares from all training dates with stronger smoothing, chosen on the inner
# validation window from windows {14, 28, 42, 56, all} x smoothing {0.5, 1, 3, 10}; plus a level
# calibration factor = inner-validation actual / predicted category units. Both use training dates only.
V2_SHARE_SMOOTHING = 10.
V2_METHOD = 'regularized_poisson_v2'


def daily_panel(raw, catalog, dates):
    base = pd.DataFrame({'Date': dates}).merge(catalog[['Branch', 'Category']].drop_duplicates(), how='cross')
    actual = raw.groupby(['Date', 'Branch', 'Category']).Quantity.sum().rename('actual').reset_index()
    return base.merge(actual, on=['Date', 'Branch', 'Category'], how='left').fillna({'actual': 0}).sort_values(['Date', 'Branch', 'Category']).reset_index(drop=True)


def features(frame, start, categories):
    x = frame[['Date', 'Branch', 'Category']].copy()
    weekday = x.Date.dt.dayofweek.astype(str)
    x['series'] = x.Branch+' / '+x.Category
    x['category_weekday'] = x.Category+' / '+weekday
    x['branch_weekday'] = x.Branch+' / '+weekday
    x['trend'] = (x.Date-pd.Timestamp(start)).dt.days/90.
    numeric = ['trend']
    for cat in categories:
        name = 'trend_'+cat
        x[name] = x.trend*x.Category.eq(cat)
        numeric.append(name)
    return x[CATEGORICAL+numeric], numeric


def fit_engine(train, alpha, start, categories):
    x, numeric = features(train, start, categories)
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                             ('num', StandardScaler(), numeric)])
    model = make_pipeline(pre, PoissonRegressor(alpha=alpha, max_iter=3000, tol=1e-7))
    model.fit(x, train.actual)
    return model


def fit_baseline(train):
    # Seasonal counts and weekday-specific anchor dates use training data only.
    latest = train.Date.max()
    past = train[train.Date.gt(latest-pd.Timedelta(days=28))].copy()
    past['weekday'] = past.Date.dt.dayofweek
    past['day_number'] = (past.Date-train.Date.min()).dt.days
    table = past.groupby(['Branch', 'Category', 'weekday']).agg(mean=('actual', 'mean'), anchor=('day_number', 'mean')).reset_index()
    network = train.groupby('Date').actual.sum()
    t = (network.index-train.Date.min()).days.to_numpy()/90.
    design = np.column_stack([np.ones(len(t)), t]+[(network.index.dayofweek == k).astype(float) for k in range(1, 7)])
    beta = np.linalg.lstsq(design, np.log(np.maximum(network.to_numpy(), .1)), rcond=None)[0]
    return dict(table=table, daily_log_growth=float(beta[1]/90.))


def baseline_predict(frame, baseline, start, growth=False):
    x = frame[['Date', 'Branch', 'Category']].copy()
    x['weekday'] = x.Date.dt.dayofweek
    x = x.merge(baseline['table'], on=['Branch', 'Category', 'weekday'], how='left', validate='many_to_one')
    if x['mean'].isna().any():
        raise ValueError('Unseen category or branch requires a cold-start rule')
    values = x['mean'].to_numpy()
    if growth:
        elapsed = (x.Date-pd.Timestamp(start)).dt.days.to_numpy()-x.anchor.to_numpy()
        values = values*np.exp(baseline['daily_log_growth']*elapsed)
    return np.maximum(values, 1e-8)


def sku_parameters(train):
    t = train.copy()
    t['cost_total'] = t.Quantity*t['Unit Cost (USD)']
    econ = t.groupby(['Branch', 'Category', 'Product']).agg(units=('Quantity', 'sum'), revenue=('Line Revenue (USD)', 'sum'), cost_total=('cost_total', 'sum')).reset_index()
    econ['price'] = econ.revenue/econ.units
    econ['cost'] = econ.cost_total/econ.units
    recent = t[t.Date.gt(t.Date.max()-pd.Timedelta(days=28))].groupby(['Branch', 'Product']).Quantity.sum().rename('recent_units').reset_index()
    econ = econ.merge(recent, on=['Branch', 'Product'], how='left').fillna({'recent_units': 0})
    groups = econ.groupby(['Branch', 'Category'])
    econ['item_share'] = (econ.recent_units+1)/(groups.recent_units.transform('sum')+groups.Product.transform('size'))
    econ['item_share_v2'] = (econ.units+V2_SHARE_SMOOTHING)/(groups.units.transform('sum')+V2_SHARE_SMOOTHING*groups.Product.transform('size'))
    if 'Unit Price (USD)' in t:
        list_price = t.groupby(['Branch', 'Product'])['Unit Price (USD)'].max().rename('list_price').reset_index()
        econ = econ.merge(list_price, on=['Branch', 'Product'], how='left')
    early = t[t.Time.str[:2].astype(int).lt(16)].groupby(['Branch', 'Category']).Quantity.sum().rename('early_units').reset_index()
    total = t.groupby(['Branch', 'Category']).Quantity.sum().rename('total_units').reset_index().merge(early, on=['Branch', 'Category'], how='left').fillna({'early_units': 0})
    total['early_share'] = total.early_units/total.total_units
    return econ.merge(total[['Branch', 'Category', 'early_share']], on=['Branch', 'Category'])


def metric(y, mean):
    y, mean = np.asarray(y), np.maximum(np.asarray(mean), 1e-8)
    return dict(wape_pct=float(100*np.abs(y-mean).sum()/max(y.sum(), 1)), mae_units=float(np.abs(y-mean).mean()),
                bias_pct=float(100*(mean-y).sum()/max(y.sum(), 1)), poisson_deviance=float(mean_poisson_deviance(y, mean)))


def share_column(method):
    return 'item_share_v2' if method == V2_METHOD else 'item_share'


def category_mean(artifact, frame, method=None):
    """Frozen category-level forecast for any method stored in the artifact."""
    method = method or artifact['selected_method']
    if method in ('regularized_poisson', V2_METHOD):
        pred = artifact['model'].predict(features(frame, artifact['start'], artifact['categories'])[0])
        return pred*artifact.get('v2_calibration', 1.) if method == V2_METHOD else pred
    return baseline_predict(frame, artifact['baseline'], artifact['start'], method == 'seasonal_growth')


def main():
    path = ROOT/'data/stories_transactions.csv'
    raw = pd.read_csv(path, parse_dates=['Date'])
    dates = pd.date_range(raw.Date.min(), raw.Date.max())
    cut = len(dates)//2
    train_dates, test_dates = dates[:cut], dates[cut:]
    training = raw[raw.Date.isin(train_dates)].copy()
    testing = raw[raw.Date.isin(test_dates)].copy()
    catalog = training[['Branch', 'Category', 'Product']].drop_duplicates()
    unseen = testing.merge(catalog, on=['Branch', 'Category', 'Product'], how='left', indicator=True)
    if unseen._merge.eq('left_only').any():
        raise ValueError('Test includes unseen products: document cold-start handling before scoring')
    start = str(train_dates.min().date())
    categories = sorted(catalog.Category.unique())
    outer_train = daily_panel(training, catalog, train_dates)
    outer_test = daily_panel(testing, catalog, test_dates)
    inner_cut = train_dates[-21]
    inner_train = outer_train[outer_train.Date.lt(inner_cut)]
    inner_valid = outer_train[outer_train.Date.ge(inner_cut)]
    inner_baseline = fit_baseline(inner_train)
    tuning = []
    for alpha in ALPHAS:
        fitted = fit_engine(inner_train, alpha, start, categories)
        pred = fitted.predict(features(inner_valid, start, categories)[0])
        tuning.append(dict(method='regularized_poisson', alpha=alpha, **metric(inner_valid.actual, pred)))
    chosen = min(tuning, key=lambda r: r['poisson_deviance'])
    best_alpha = chosen['alpha']
    for growth in [False, True]:
        pred = baseline_predict(inner_valid, inner_baseline, start, growth)
        tuning.append(dict(method='seasonal_growth' if growth else 'seasonal_frozen', alpha=None, **metric(inner_valid.actual, pred)))
    validation_winner = min(tuning, key=lambda r: r['poisson_deviance'])['method']
    # v2: calibration and item-share choice use the inner validation window only (no April-June data).
    inner_model = fit_engine(inner_train, best_alpha, start, categories)
    inner_pred = inner_model.predict(features(inner_valid, start, categories)[0])
    v2_calibration = float(inner_valid.actual.sum()/inner_pred.sum())
    inner_params = sku_parameters(training[training.Date.lt(inner_cut)])
    inner_obs = training[training.Date.ge(inner_cut)].groupby(['Date', 'Branch', 'Product']).Quantity.sum().rename('actual').reset_index()
    inner_cat = inner_valid[['Date', 'Branch', 'Category']].assign(pred=inner_pred)
    inner_item = inner_cat.merge(inner_params, on=['Branch', 'Category']).merge(inner_obs, on=['Date', 'Branch', 'Product'], how='left').fillna({'actual': 0})
    v2_inner = dict(original=metric(inner_item.actual, inner_item.pred*inner_item.item_share),
                    v2=metric(inner_item.actual, inner_item.pred*v2_calibration*inner_item.item_share_v2))
    if v2_inner['v2']['poisson_deviance'] < v2_inner['original']['poisson_deviance'] and validation_winner == 'regularized_poisson':
        validation_winner = V2_METHOD
    model = fit_engine(outer_train, best_alpha, start, categories)
    baseline = fit_baseline(outer_train)
    predictions = dict(regularized_poisson=model.predict(features(outer_test, start, categories)[0]),
                       seasonal_frozen=baseline_predict(outer_test, baseline, start, False),
                       seasonal_growth=baseline_predict(outer_test, baseline, start, True))
    predictions[V2_METHOD] = predictions['regularized_poisson']*v2_calibration
    params = sku_parameters(training)
    comparisons, all_sku = [], []
    for name, pred in predictions.items():
        comparisons.append(dict(method=name, level='branch_category', **metric(outer_test.actual, pred)))
        cat = outer_test[['Date', 'Branch', 'Category']].copy()
        cat['predicted_category_units'] = pred
        item = cat.merge(params[['Branch', 'Category', 'Product', share_column(name), 'price', 'cost', 'early_share']].rename(columns={share_column(name): 'item_share'}), on=['Branch', 'Category'], validate='many_to_many')
        observed = testing.groupby(['Date', 'Branch', 'Product']).Quantity.sum().rename('actual').reset_index()
        item = item.merge(observed, on=['Date', 'Branch', 'Product'], how='left').fillna({'actual': 0})
        item['predicted_mean_units'] = item.predicted_category_units*item.item_share
        item['method'] = name
        for scope, subset in [('all_sku', item), ('perishable_sku', item[item.Category.isin(PERISHABLES)])]:
            comparisons.append(dict(method=name, level=scope, **metric(subset.actual, subset.predicted_mean_units)))
        # One-batch service-aware scenario. This is not the observed company policy.
        stock = []
        for mu, p, c in zip(item.predicted_mean_units, item.price, item.cost):
            unconstrained = int(poisson.ppf(np.clip((p-c)/p, 0., .999999), mu))
            stock.append(max(unconstrained, fill_floor_stock(mu, .90)))
        item['prepare_units'] = stock
        item['sold_units'] = np.minimum(item.actual, item.prepare_units)
        item['unsold_units'] = item.prepare_units-item.sold_units
        item['unmet_units'] = item.actual-item.sold_units
        item['simulated_contribution'] = item.price*item.sold_units-item.cost*item.prepare_units
        all_sku.append(item)
    detail = pd.concat(all_sku, ignore_index=True)
    detail.to_csv(OUT/'strict_half_predictions.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'strict_half_metrics.csv', index=False)
    perishable = detail[detail.Category.isin(PERISHABLES)]
    finance = perishable.groupby('method')[['actual','prepare_units','sold_units','unsold_units','unmet_units','simulated_contribution']].sum().reset_index()
    finance['fill_rate_pct'] = 100*finance.sold_units/finance.actual
    finance.to_csv(OUT/'strict_half_finance.csv', index=False)
    daily_profit = perishable.groupby(['Date', 'method']).simulated_contribution.sum().unstack()
    rng = np.random.default_rng(490)
    uncertainty = []
    for leader, comparator in [('regularized_poisson', 'seasonal_growth'), ('regularized_poisson', 'seasonal_frozen'),
                               (V2_METHOD, 'regularized_poisson'), (V2_METHOD, 'seasonal_frozen')]:
        delta = (daily_profit[leader]-daily_profit[comparator]).to_numpy()
        count = len(delta)
        starts = rng.integers(0, count, size=(3000, int(np.ceil(count/7))))
        indices = (starts[:, :, None]+np.arange(7)[None, None, :]) % count
        draws = delta[indices.reshape(3000, -1)[:, :count]].sum(axis=1)
        uncertainty.append(dict(comparison=leader+' minus '+comparator,
            contribution_difference_usd=float(delta.sum()),
            conditional_circular_7day_block_interval_95=[float(v) for v in np.quantile(draws, [.025, .975])],
            warning='Retrospective uncertainty conditional on fixed forecasts and hypothetical inventory costs; not future ROI or uncertainty over missing operations.'))
    artifact = dict(model=model, start=start, categories=categories, catalog=catalog, sku_parameters=params,
                    baseline=baseline, selected_alpha=best_alpha, selected_method=validation_winner,
                    v2_calibration=v2_calibration,
                    train_end=str(train_dates.max().date()), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(artifact, ROOT/'models/strict_half_model.joblib')
    result = dict(protocol='Frozen chronological half/half, entire test period predicted at the training cutoff',
        input_files=['data/stories_transactions.csv'], external_model_inputs=False,
        train=dict(start=start, end=str(train_dates.max().date()), days=len(train_dates), rows=len(training)),
        test=dict(start=str(test_dates.min().date()), end=str(test_dates.max().date()), days=len(test_dates), rows=len(testing)),
        inner_validation=dict(start=str(inner_cut.date()), end=str(train_dates.max().date()), days=21),
        chosen_alpha=best_alpha, selected_method_on_inner_validation=validation_winner, tuning=tuning,
        v2=dict(method=V2_METHOD, description='Same engine and alpha; item shares from all training dates with smoothing 10; '
                'category level multiplied by the inner-validation calibration factor (actual/predicted).',
                calibration_factor=v2_calibration, share_smoothing=V2_SHARE_SMOOTHING,
                inner_validation_sku=v2_inner, selection_rule='Use v2 if its inner-validation SKU Poisson deviance beats the '
                'original allocation; chosen before scoring April-June.'),
        metrics=comparisons, perishable_one_batch_scenario=finance.to_dict('records'),
        conditional_contribution_uncertainty=uncertainty,
        feature_contract=['Only branch/category/product identifiers and dates from the transaction file.',
                          'Date-derived weekdays and trend; item shares, costs, prices and early shares frozen from training.',
                          'No test sales lags, no test-period refits, no weather, holidays, conflict, news or external files.'],
        limitation='Dataset already explored before this protocol was requested. This is a clean computational boundary, not a newly untouched test set.',
        financial_assumptions=['Sales treated as demand; zero salvage; unit costs/paid prices from training only; no capacity, substitution or labor costs.',
                               'Single-batch 90% expected unit-fill floor; demand Poisson; hypothetical preparation, not measured current practice.'],
        artifact='models/strict_half_model.joblib', input_sha256=artifact['input_sha256'])
    (OUT/'strict_half_results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['train','test','chosen_alpha','selected_method_on_inner_validation']}, indent=2), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    print(finance.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
