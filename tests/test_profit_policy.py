import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from profit_policy import expected_sold, optimize, realize


class ProfitPolicyTests(unittest.TestCase):
    def test_expected_sales_against_direct_probability_sum(self):
        d = np.arange(200)
        for mean in (0., .2, 3., 25.):
            for stock in (0, 1, 4, 30):
                exact = np.dot(np.minimum(d, stock), poisson.pmf(d, mean))
                self.assertAlmostEqual(float(expected_sold(mean, stock)), exact, places=9)

    def test_stock_and_cash_conservation(self):
        policy = optimize(4., .6, 6., 2., setup=.4, premium=.2, salvage_fraction=.3)
        for early in range(10):
            for late in range(10):
                r = realize(policy, early, late)
                self.assertEqual(r['opening_units']+r['replenished_units'], r['sold_units']+r['unsold_units'])
                self.assertEqual(early+late, r['sold_units']+r['unmet_units'])
                cash = 6*r['sold_units']+.6*r['unsold_units']-2*r['opening_units']-2.2*r['replenished_units']-.4*r['replenishment_executed']
                self.assertAlmostEqual(r['contribution'], cash)

    def test_dynamic_program_against_independent_enumeration(self):
        # Small unconstrained case: enumerate demands and both decisions directly.
        policy = optimize(2., .5, 5., 2., setup=.3, premium=.1, fill_floor=0.)
        demands = np.arange(25)
        probs = poisson.pmf(demands, 1.)
        first_values = []
        for opening in range(13):
            value = -2*opening
            for early, prob in zip(demands, probs):
                remaining = max(opening-early, 0)
                choices = []
                for topup in range(13):
                    revenue = 5*np.dot(np.minimum(demands, remaining+topup), probs)
                    choices.append(revenue-2.1*topup-.3*(topup > 0))
                value += prob*(5*min(early, opening)+max(choices))
            first_values.append(value)
        self.assertEqual(policy['opening_units'], int(np.argmax(first_values)))
        self.assertAlmostEqual(policy['expected_two_stage_contribution'], max(first_values), places=6)

    def test_free_flexibility_dominates_without_service_constraint(self):
        for mean in (.5, 3., 10.):
            p = optimize(mean, .4, 6., 2., setup=0., premium=0., fill_floor=0.)
            self.assertGreaterEqual(p['expected_two_stage_contribution']+1e-6, p['expected_one_batch_contribution'])

    def test_zero_demand_and_invalid_inputs(self):
        p = optimize(0., .5, 6., 2.)
        self.assertEqual(p['opening_units'], 0)
        self.assertEqual(realize(p, 0, 0)['contribution'], 0)
        for mean in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                optimize(mean, .5, 6., 2.)
        with self.assertRaises(ValueError):
            realize(p, -1, 0)

    def test_replenishment_does_not_observe_late_demand(self):
        p = optimize(4., .5, 6., 2.)
        self.assertEqual(realize(p, 1, 0)['replenished_units'], realize(p, 1, 50)['replenished_units'])


if __name__ == '__main__':
    unittest.main()
