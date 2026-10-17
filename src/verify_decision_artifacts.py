"""Reconcile persisted replay artifacts and record their source/version provenance."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    detail = pd.read_csv(ROOT/'reports/two_stage_replay.csv')
    summary = pd.read_csv(ROOT/'reports/two_stage_summary.csv')
    assert len(detail) == 20520
    assert not detail.duplicated(['Date', 'Branch', 'Product', 'scenario', 'forecast']).any()
    assert np.allclose(detail.opening_units+detail.replenished_units, detail.sold_units+detail.unsold_units)
    assert np.allclose(detail.actual, detail.sold_units+detail.unmet_units)
    assert np.allclose(detail.one_batch_units, detail.one_batch_sold+detail.one_batch_unsold)
    assert np.allclose(detail.actual, detail.one_batch_sold+detail.one_batch_unmet)
    for row in summary.itertuples():
        group = detail[detail.scenario.eq(row.scenario)&detail.forecast.eq(row.forecast)]
        assert len(group) == 1710
        for name in ['contribution', 'one_batch_contribution', 'incremental_contribution', 'actual', 'sold_units', 'unsold_units']:
            assert np.isclose(group[name].sum(), getattr(row, name)), name
    cards = json.loads((ROOT/'reports/decision_cards.json').read_text(encoding='utf-8'))
    assert len(cards) == 114
    assert {c['forecast'] for c in cards} == {'strong_baseline', 'internal'}
    forbidden = {'actual', 'sold_units', 'unmet_units', 'early', 'late', 'contribution'}
    assert all(not forbidden.intersection(c) for c in cards)
    assert all('Historical' in c['status'] for c in cards)
    raw = ROOT/'data/stories_transactions.csv'
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    prior = json.loads((ROOT/'reports/run_manifest.json').read_text(encoding='utf-8'))
    assert raw_hash == prior['input_sha256']['data/stories_transactions.csv'], 'Raw input changed'
    paths = ['data/stories_transactions.csv', 'data/external/outlier_event_registry.json',
             'data/external/macro_context_audit.json', 'src/profit_policy.py', 'src/outlier_forensics.py',
             'src/recommend.py', 'src/demand_model.py', 'reports/two_stage_replay.csv',
             'reports/two_stage_results.json', 'reports/decision_cards.json', 'tests/test_profit_policy.py']
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['pandas','numpy','scipy','scikit-learn','statsmodels','matplotlib']},
        sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        assertions_passed=['20520 unique scenario/forecast/item/day replay records',
                           'Inventory and demand conservation for every replay record',
                           'Summary contribution and counts reconcile to detailed replay',
                           '114 historical cards include both forecast candidates and exclude target outcomes',
                           'Raw transaction hash unchanged from original analysis'],
        evaluation_status='Retrospective policy replay; June previously inspected, not independent confirmation',
        test_command='python -m unittest discover -s tests -v (run separately)')
    (ROOT/'reports/decision_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['assertions_passed'], indent=2))


if __name__ == '__main__':
    main()
