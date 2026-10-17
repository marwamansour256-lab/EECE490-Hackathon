"""Reconcile the expanded context models, daily dossiers and source coverage."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import subprocess
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=ROOT/'data/external/context_v2'
    source=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    assert len(source['records'])==9
    for r in source['records']:
        assert r['status']=='ok' and r['hourly_count']==4752
        assert all(v['null_count']==0 for v in r['fields'].values())
        assert hashlib.sha256((cache/(r['name']+'.json')).read_bytes()).hexdigest()==r['sha256']
    context=pd.read_csv(cache/'daily_features.csv')
    assert len(context)==543 and not context.duplicated(['Date','region']).any()
    assert set(context.region)=={'beirut','batroun','broummana'}
    assert not context.select_dtypes('number').isna().any().any()
    for prefix in ['context_v2','context_v2_orders']:
        predictions=pd.read_csv(ROOT/('reports/'+prefix+'_predictions.csv'))
        assert not predictions.duplicated(['Date','Branch','Category','method']).any()
        assert (predictions.context_mean>=0).all() and predictions.actual.ge(0).all()
        parts=pd.read_csv(ROOT/('reports/'+prefix+'_decomposition.csv'))
        parts=parts.groupby(['Date','Branch','Category','method']).contribution_units.sum().rename('adjustment').reset_index()
        joined=predictions.merge(parts,on=['Date','Branch','Category','method'],validate='one_to_one')
        assert len(joined)==len(predictions)
        np.testing.assert_allclose(joined.baseline_mean+joined.adjustment,joined.context_mean,atol=1e-7)
        trained=joblib.load(ROOT/('models/'+prefix+'_model.joblib'))
        assert trained['training_end']=='2026-03-31'
        for key,value in trained['models'].items():
            if key.startswith('selection_'):continue
            if value['family']!='observed_oracle':
                assert not any(c.split(' / ')[0] in ['rain_sum','apparent_mean','dust_mean','pm25_mean','rain_forecast_surprise','temperature_forecast_surprise'] for c in value['columns'])
    mix=pd.read_csv(ROOT/'reports/context_v2_mix_predictions.csv')
    assert len(mix)==4*91*135
    assert not mix.duplicated(['Date','Branch','Product','method']).any()
    np.testing.assert_allclose(mix.prepare_units,mix.sold_units+mix.unsold_units)
    np.testing.assert_allclose(mix.Quantity,mix.sold_units+mix.unmet_units)
    np.testing.assert_allclose(mix.contribution,mix.price*mix.sold_units-mix.cost*mix.prepare_units)
    sums=mix.groupby(['Date','Branch','Category','method']).predicted_mean_units.sum()
    totals=mix.groupby(['Date','Branch','Category','method']).category_mean.first()
    np.testing.assert_allclose(sums,totals)
    ledger=json.loads((ROOT/'reports/daily_why_ledger.json').read_text(encoding='utf-8'))
    assert len(ledger['records'])==724
    keys={(r['date'],r['branch']) for r in ledger['records']}
    assert len(keys)==724 and len({r['date'] for r in ledger['records']})==181
    for r in ledger['records']:
        bridge=sum(r['exact_accounting_bridge_usd'].values())
        assert np.isclose(bridge,r['accounting_profit_gap_usd'])
        assert np.isclose(r['accounting_baseline']['gross_profit_usd']+bridge,r['actual']['recorded_gross_profit_usd'])
        forecast=r['forecast']
        assert np.isclose(forecast['baseline_units']+sum(forecast['model_adjustment_by_group_units'].values()),forecast['predicted_units'])
        assert np.isclose(forecast['predicted_units']+forecast['unexplained_actual_minus_prediction_units'],r['actual']['units'])
        assert r['event_and_mechanism_evidence']
    union=pd.read_csv(ROOT/'reports/context_v2_all_associations.csv')
    assert len(union)==2337 and union.p_value.notna().sum()==2277
    assert union.q_global_bh.notna().sum()==2277
    internet=pd.read_csv(cache/'internet_daily.csv')
    assert len(internet)==181 and internet.coverage_fraction.min()>=.9
    assert internet.reachability_min_drop_proxy_pct.isna().any()
    raw_hash=hashlib.sha256((ROOT/'data/stories_transactions.csv').read_bytes()).hexdigest()
    old=json.loads((ROOT/'reports/run_manifest.json').read_text(encoding='utf-8'))
    assert raw_hash==old['input_sha256']['data/stories_transactions.csv']
    paths=['data/stories_transactions.csv','data/external/context_v2/manifest.json','data/external/context_v2/internet_manifest.json',
        'data/external/external_events.json','src/context_features.py','src/context_model.py','src/context_mix_model.py',
        'src/context_associations.py','src/context_robustness.py','src/day_explanations.py','src/internet_checks.py',
        'reports/context_v2_model_results.json','reports/context_v2_orders_model_results.json','reports/context_v2_mix_results.json',
        'reports/context_v2_all_associations.csv','reports/daily_why_ledger.json','models/context_v2_model.joblib','models/context_v2_orders_model.joblib','models/context_v2_mix_model.joblib']
    checks=['Nine complete cached hourly streams and543 unique region/date feature rows',
        'All category-volume and order-traffic decompositions reconcile to predictions',
        'Predictive models exclude observed target-day weather, air and forecast surprises',
        '49140 item/day/method forecasts conserve category volumes, stocks, demand and cash',
        '724 unique daily scope dossiers reconcile financial bridges and explicitly unexplained demand',
        'Global2337-row search correction includes the added internet hypotheses',
        'Internet coverage tracked separately; unavailable antecedent signal not turned into an outage',
        'Raw Stories transaction hash unchanged; no external sales labels']
    manifest=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ['pandas','numpy','scipy','scikit-learn','statsmodels','joblib']},
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},checks_passed=checks,
        evaluation_status='Exploratory extension designed after inspecting the dataset; not new independent confirmation or measured company ROI')
    (ROOT/'reports/context_v2_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
