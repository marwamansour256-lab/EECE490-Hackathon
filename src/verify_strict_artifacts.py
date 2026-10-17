"""Independently reconcile the frozen half/half artifact and its saved evaluation."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import importlib.metadata
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from strict_model import features, V2_METHOD
from provenance import matches_recorded_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/'reports/strict_half_results.json').read_text(encoding='utf-8'))
    artifact = joblib.load(ROOT/'models/strict_half_model.joblib')
    raw = ROOT/'data/stories_transactions.csv'
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result['input_sha256'] == artifact['input_sha256']
    assert matches_recorded_sha256(raw,result['input_sha256'])
    assert result['input_files'] == ['data/stories_transactions.csv'] and result['external_model_inputs'] is False
    assert result['train']['end'] == artifact['train_end'] == '2026-03-31'
    assert result['test']['start'] == '2026-04-01' and result['test']['days'] == 91
    assert result['inner_validation']['end'] <= result['train']['end']
    detail = pd.read_csv(ROOT/'reports/strict_half_predictions.csv', parse_dates=['Date'])
    assert len(detail) == 4*91*135
    assert not detail.duplicated(['Date', 'Branch', 'Product', 'method']).any()
    assert detail.Date.min() == pd.Timestamp('2026-04-01')
    assert detail.Date.max() == pd.Timestamp('2026-06-30')
    assert np.allclose(detail.actual, detail.sold_units+detail.unmet_units)
    assert np.allclose(detail.prepare_units, detail.sold_units+detail.unsold_units)
    assert np.allclose(detail.simulated_contribution, detail.price*detail.sold_units-detail.cost*detail.prepare_units)
    sums = detail.groupby(['Date', 'Branch', 'Category', 'method']).predicted_mean_units.sum()
    categories = detail.groupby(['Date', 'Branch', 'Category', 'method']).predicted_category_units.first()
    np.testing.assert_allclose(sums, categories)
    # Check stored ML forecasts against fresh predictions from the serialized model.
    ml = detail[detail.method.eq('regularized_poisson')].drop_duplicates(['Date','Branch','Category'])
    frame = ml[['Date','Branch','Category']]
    expected = artifact['model'].predict(features(frame, artifact['start'], artifact['categories'])[0])
    np.testing.assert_allclose(expected, ml.predicted_category_units)
    # v2 = the same serialized model times the inner-validation calibration factor.
    v2 = detail[detail.method.eq(V2_METHOD)].drop_duplicates(['Date','Branch','Category'])
    np.testing.assert_allclose(artifact['model'].predict(features(v2[['Date','Branch','Category']], artifact['start'],
                               artifact['categories'])[0])*artifact['v2_calibration'], v2.predicted_category_units)
    assert abs(result['v2']['calibration_factor']-artifact['v2_calibration']) < 1e-12
    # All learned economics and item shares must remain fixed through test dates.
    for field in ['price', 'cost', 'item_share', 'early_share']:
        assert detail.groupby(['method','Branch','Product'])[field].nunique().max() == 1
    files = ['data/stories_transactions.csv', 'src/strict_model.py', 'src/predict_strict.py',
             'models/strict_half_model.joblib', 'reports/strict_half_results.json',
             'reports/strict_half_predictions.csv', 'tests/test_strict_model.py']
    checks = ['Single transaction CSV raw input; no external predictive files',
              'Frozen March 31 artifact and disjoint April-June test dates',
              '49140 unique item-day-method predictions (4 methods incl. regularized_poisson_v2); 91 dates and 135 branch-items per method',
              'Category-to-item forecasts reconcile; serialized model reproduces saved forecasts, including v2 calibration',
              'Demand, stock and contribution reconcile for every saved record',
              'Item shares, price, cost and intraday proportions remain constant through testing']
    manifest = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['pandas','numpy','scipy','scikit-learn','joblib']},
        sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        checks_passed=checks, evaluation_status=result['limitation'])
    (ROOT/'reports/strict_half_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
