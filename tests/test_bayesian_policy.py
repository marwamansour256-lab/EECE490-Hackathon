import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bayesian_policy import joint_demand, conditional_late, posterior_parameters, optimize_adaptive, realize_adaptive
from profit_policy import optimize


class BayesianPolicyTests(unittest.TestCase):
    def test_joint_has_correct_marginal_means_and_covariance(self):
        joint, omitted = joint_demand(2., 3., shape=5.)
        early = np.arange(joint.shape[0])
        late = np.arange(joint.shape[1])
        self.assertLess(omitted, 5e-9)
        self.assertAlmostEqual(float(early@joint.sum(axis=1)), 2., places=6)
        self.assertAlmostEqual(float(late@joint.sum(axis=0)), 3., places=6)
        covariance = float(np.sum(joint*(early[:, None]-2)*(late[None, :]-3)))
        self.assertAlmostEqual(covariance, 2*3/5, places=5)

    def test_exact_posterior_matches_conjugate_formula(self):
        joint, _ = joint_demand(2., 3., shape=5.)
        pmf = conditional_late(joint, 4)
        mean, shape = posterior_parameters(2., 3., 4, 5.)
        self.assertEqual(shape, 9.)
        self.assertAlmostEqual(float(np.arange(len(pmf))@pmf), mean, places=5)

    def test_sold_out_is_lower_bound_not_exact_demand(self):
        joint, _ = joint_demand(2., 3., shape=2.)
        exact = conditional_late(joint, 2, sold_out=False)
        censored = conditional_late(joint, 2, sold_out=True)
        demand = np.arange(joint.shape[1])
        self.assertGreater(float(demand@censored), float(demand@exact))
        np.testing.assert_allclose(conditional_late(joint, 0, sold_out=True), joint.sum(axis=0), atol=1e-12)

    def test_poisson_control_matches_existing_optimizer(self):
        for mean in (0., 2., 5.):
            old = optimize(mean, .4, 6., 2.)
            new = optimize_adaptive(mean, .4, 6., 2., shape=None)
            self.assertEqual(old['opening_units'], new['opening_units'])
            self.assertEqual(old['one_batch_units'], new['one_batch_units'])
            self.assertAlmostEqual(old['expected_two_stage_contribution'], new['expected_two_stage_contribution'], places=5)
            for remaining in range(new['opening_units']+1):
                self.assertEqual(old['replenish_by_remaining_stock'][remaining], new['replenish_by_remaining_stock'][str(remaining)])

    def test_censoring_replay_cannot_see_lost_or_future_demand(self):
        policy = optimize_adaptive(4., .5, 6., 2., shape=5.)
        q = policy['opening_units']
        first = realize_adaptive(policy, q, 0)
        second = realize_adaptive(policy, q+50, 80)
        self.assertEqual(first['replenished_units'], second['replenished_units'])
        for early, late in [(0, 0), (2, 3), (q+50, 80)]:
            r = realize_adaptive(policy, early, late)
            self.assertEqual(r['opening_units']+r['replenished_units'], r['sold_units']+r['unsold_units'])
            self.assertEqual(early+late, r['sold_units']+r['unmet_units'])

    def test_static_ablation_does_not_update_from_early_count(self):
        joint, _ = joint_demand(2., 3., shape=5., adaptive=False)
        np.testing.assert_allclose(conditional_late(joint, 0), conditional_late(joint, 5), atol=1e-12)

    def test_information_ablation_keeps_same_one_batch_prior(self):
        adaptive = optimize_adaptive(4., .5, 6., 2., shape=3., fill_floor=0.)
        frozen = optimize_adaptive(4., .5, 6., 2., shape=3., fill_floor=0., use_observation=False)
        self.assertEqual(adaptive['one_batch_units'], frozen['one_batch_units'])
        self.assertAlmostEqual(adaptive['expected_one_batch_contribution'], frozen['expected_one_batch_contribution'])
        self.assertGreaterEqual(adaptive['expected_two_stage_contribution']+1e-8, frozen['expected_two_stage_contribution'])


if __name__ == '__main__':
    unittest.main()
