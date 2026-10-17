import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from growth_plan import prep_card,profit_scenario


class GrowthPlanTests(unittest.TestCase):
    def test_incremental_profit_is_explicit_arithmetic_not_model_score(self):
        result=profit_scenario(1000,.03,3.5,.5,waste=5,labor=10,implementation=5)
        self.assertEqual(result["additional_add_ons"],30)
        self.assertEqual(result["incremental_contribution_scenario_usd"],70)
        self.assertAlmostEqual(result["break_even_incremental_acceptance"],20/3000)
        self.assertEqual(profit_scenario(100,0,4)["incremental_contribution_scenario_usd"],0)

    def test_invalid_money_inputs_fail(self):
        for rate in [-.1,1.1,float("nan")]:
            with self.assertRaises(ValueError):
                profit_scenario(100,rate,3)
        with self.assertRaises(ValueError):
            profit_scenario(100,.1,3,labor=-1)

    def test_prep_bridge_reproduces_committed_model_example(self):
        card=prep_card("2026-06-30","Harbour","Turkey & Cheese Sandwich")
        # Committed artifact selects regularized_poisson_v2 (calibrated level, full-training item shares).
        self.assertEqual(card["method"],"regularized_poisson_v2")
        self.assertAlmostEqual(card["forecast_mean_units"],3.669136909412762,places=7)
        self.assertEqual(card["opening_prepare_units"],4)
        self.assertEqual(card["at16_topup_by_remaining_stock"]["0"],4)
        self.assertFalse(card["test_sales_used_in_forecast"])
        self.assertFalse(card["recommendation_uplift_added_to_forecast"])
        drink=prep_card("2026-06-30","Harbour","Iced Matcha")
        self.assertFalse(drink["preparation_applicable"])
        self.assertNotIn("opening_prepare_units",drink)


if __name__=="__main__":
    unittest.main()
