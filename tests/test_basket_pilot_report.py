from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from basket_pilot_report import REQUIRED,validate,summarize


def row(order="a",arm="current",date="2026-07-01"):
    return dict(experiment_id="pilot",experiment_type="recommendation",date=date,branch="A",arm=arm,
        order_id=order,eligible=1,offer_shown=int(arm!="current"),accepted=0,stock_available=1,
        order_revenue_usd=10.,order_cost_usd=4.,discount_usd=2.,waste_cost_allocated_usd=0.,
        extra_labor_allocated_usd=0.,other_cost_allocated_usd=0.,assignment_probability=.5)


class BasketPilotTests(unittest.TestCase):
    def test_net_revenue_does_not_have_discounts_subtracted_twice(self):
        result=validate(pd.DataFrame([row()]))
        self.assertEqual(result.contribution_usd.iloc[0],6.)
        r=row();r["extra_labor_allocated_usd"]=1;r["waste_cost_allocated_usd"]=2
        self.assertEqual(validate(pd.DataFrame([r])).contribution_usd.iloc[0],3.)

    def test_duplicate_orders_and_mixed_assignment_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate(pd.DataFrame([row(),row()]))
        with self.assertRaisesRegex(ValueError,"multiple arms"):
            validate(pd.DataFrame([row(),row("b","model")]))

    def test_complete_order_denominator_and_declines_are_retained(self):
        rows=[row("a","current"),row("b","current"),row("c","model","2026-07-02")]
        rows[2]["order_revenue_usd"]=12
        result=summarize(pd.DataFrame(rows),draws=50)["experiments"][0]
        self.assertEqual(result["arms"]["current"]["eligible_orders"],2)
        self.assertEqual(result["arms"]["model"]["observed_acceptance_per_shown"],0)
        self.assertEqual(result["comparisons"][0]["observed_contribution_difference_per_eligible_order_usd"],2)

    def test_missing_measurements_and_impossible_acceptance_fail(self):
        r=row();r["order_cost_usd"]=None
        with self.assertRaises(ValueError):
            validate(pd.DataFrame([r]))
        r=row();r["accepted"]=1
        with self.assertRaisesRegex(ValueError,"shown offer"):
            validate(pd.DataFrame([r]))

    def test_empty_template_does_not_invent_roi(self):
        result=summarize(pd.DataFrame(columns=REQUIRED))
        self.assertEqual(result["experiments"],[])
        self.assertFalse(result["incremental_profit_measured"])


if __name__=="__main__":
    unittest.main()
