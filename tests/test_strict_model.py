import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from strict_model import features, fit_engine, fit_baseline, baseline_predict, sku_parameters


class StrictModelTests(unittest.TestCase):
    def test_frozen_predictions_do_not_use_test_labels(self):
        dates = pd.date_range('2026-01-01', periods=90)
        train = pd.DataFrame({'Date': dates, 'Branch': 'A', 'Category': 'Bakery',
                              'actual': 10+np.arange(90)//15})
        model = fit_engine(train, .01, '2026-01-01', ['Bakery'])
        test = pd.DataFrame({'Date': pd.date_range('2026-04-01', periods=91),
                             'Branch': 'A', 'Category': 'Bakery', 'actual': 0.})
        before = model.predict(features(test, '2026-01-01', ['Bakery'])[0])
        baseline = fit_baseline(train)
        base_before = baseline_predict(test, baseline, '2026-01-01', True)
        test['actual'] = np.arange(91)*1000
        after = model.predict(features(test, '2026-01-01', ['Bakery'])[0])
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(base_before, baseline_predict(test, baseline, '2026-01-01', True))

    def test_baseline_is_frozen_without_test_lags(self):
        dates = pd.date_range('2026-01-01', periods=90)
        train = pd.DataFrame({'Date': dates, 'Branch': 'A', 'Category': 'Bakery', 'actual': 10.})
        future = pd.DataFrame({'Date': pd.date_range('2026-04-01', periods=91), 'Branch': 'A', 'Category': 'Bakery'})
        values = baseline_predict(future, fit_baseline(train), '2026-01-01', False)
        np.testing.assert_allclose(values, 10.)

    def test_training_item_shares_and_economics_reconcile(self):
        raw = pd.DataFrame([
            {'Date': pd.Timestamp('2026-03-20'), 'Time': '10:00:00', 'Branch': 'A', 'Category': 'Bakery', 'Product': 'X', 'Quantity': 2, 'Unit Cost (USD)': 2., 'Line Revenue (USD)': 10.},
            {'Date': pd.Timestamp('2026-03-21'), 'Time': '18:00:00', 'Branch': 'A', 'Category': 'Bakery', 'Product': 'Y', 'Quantity': 3, 'Unit Cost (USD)': 3., 'Line Revenue (USD)': 18.}])
        result = sku_parameters(raw).set_index('Product')
        self.assertAlmostEqual(result.item_share.sum(), 1.)
        self.assertAlmostEqual(result.loc['X', 'price'], 5.)
        self.assertAlmostEqual(result.loc['Y', 'cost'], 3.)
        self.assertAlmostEqual(result.loc['X', 'early_share'], .4)


if __name__ == '__main__':
    unittest.main()
