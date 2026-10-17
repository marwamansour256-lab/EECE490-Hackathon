import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from context_features import calendar_features,ISSUED_FORECAST,OBSERVED,CALENDAR_FORECAST
from context_model import context_design,fit_context,decompose
from context_mix_model import fit_model
from day_explanations import profit_bridge
from internet_checks import parse_signal


class ContextModelTests(unittest.TestCase):
    def test_religious_calendars_have_distinct_clock_and_exception_rules(self):
        dates=pd.to_datetime(['2026-02-20','2026-02-21','2026-02-23','2026-03-02','2026-03-19','2026-04-04','2026-06-24'])
        x=calendar_features(dates).set_index('Date')
        self.assertEqual(x.loc['2026-02-20','maronite_noon_fast'],1)
        self.assertEqual(x.loc['2026-02-21','maronite_noon_fast'],0)
        self.assertEqual(x.loc['2026-03-02','maronite_noon_fast'],0)
        self.assertEqual(x.loc['2026-04-04','maronite_noon_fast'],1)
        self.assertEqual(x.loc['2026-02-23','triple_calendar_overlap'],1)
        self.assertEqual(x.loc['2026-06-24','apostles_abstinence'],1)

    def test_external_forecast_inputs_exclude_realized_weather_and_surprise(self):
        self.assertFalse(set(ISSUED_FORECAST+CALENDAR_FORECAST).intersection(OBSERVED))
        self.assertTrue(all(c.startswith('forecast_') for c in ISSUED_FORECAST))

    def test_context_matrix_is_independent_of_sales_labels(self):
        dates=pd.date_range('2026-01-01',periods=5)
        panel=pd.DataFrame({'Date':dates,'Branch':'A','Category':'Bakery','actual':range(5)})
        ctx=pd.DataFrame({'Date':dates,'forecast_rain_sum':np.arange(5)/2})
        before,_=context_design(panel,ctx,['forecast_rain_sum'],['Bakery'])
        panel['actual']=10000
        after,_=context_design(panel,ctx,['forecast_rain_sum'],['Bakery'])
        pd.testing.assert_frame_equal(before,after)

    def test_external_adjustment_decomposition_reconciles(self):
        x=pd.DataFrame({'rain':np.arange(20)/10,'calendar':np.arange(20)%2})
        base=np.full(20,5.)
        y=np.array([4,6]*10)
        fitted=fit_context(x,y,base,.1)
        parts=decompose(fitted,x,base,{'rain':'weather','calendar':'calendar'})
        np.testing.assert_allclose(base+sum(parts.values()),base*fitted.predict(x),atol=1e-8)

    def test_accounting_bridge_is_exact_and_symmetric(self):
        before=[100,1.5,3.]
        after=[120,1.7,2.8]
        bridge=profit_bridge(before,after)
        reverse=profit_bridge(after,before)
        self.assertAlmostEqual(sum(bridge.values()),np.prod(after)-np.prod(before))
        for key in bridge:self.assertAlmostEqual(bridge[key],-reverse[key])

    def test_menu_classifier_does_not_read_product_target_as_predictor(self):
        data=pd.DataFrame({'Branch':['A','A','B','B']*10,'Product':['X','Y','X','Y']*10,
                           'weekday':['0']*40,'Quantity':[2,3,4,1]*10})
        model=fit_model(data,[],.1)
        before=model.predict_proba(data)
        data['Product']='CHANGED_TARGET_LABEL'
        np.testing.assert_array_equal(before,model.predict_proba(data))

    def test_missing_internet_signal_is_not_an_outage_value(self):
        payload={'error':None,'data':[[{'entityCode':'LB','datasource':'ping-slash24','from':1767225600,'step':3600,'values':[10,None,8]}]]}
        result=parse_signal(payload)
        self.assertTrue(pd.isna(result.reachable_blocks.iloc[1]))
        self.assertEqual(result.reachable_blocks.iloc[2],8.)


if __name__=='__main__':unittest.main()
