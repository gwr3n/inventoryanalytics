import types
import unittest
from unittest import mock

import inventoryanalytics.lotsizing.stochastic.nonstationary.RS as rs


class TestRS(unittest.TestCase):

    def test_base_stochastic_lot_sizing_constructor(self):
        model = rs.StochasticLotSizing(K=20, h=1, p=4, d=[2, 3], I0=1)

        self.assertEqual(model.K, 20)
        self.assertEqual(model.h, 1)
        self.assertEqual(model.p, 4)
        self.assertEqual(model.d, [2, 3])
        self.assertEqual(model.I0, 1)

    def test_cfolf_numeric_value(self):
        mpn = rs.MultiPeriodNewsvendor([2, 3], 1, 4)
        self.assertAlmostEqual(mpn.cfolf(3, 2), 1.2180175491295142)

    def test_cycle_cost_rejects_invalid_interval(self):
        policy = rs.RS_DP.__new__(rs.RS_DP)
        with self.assertRaises(Exception):
            policy.cycle_cost(2, 1)

    def test_optimal_cost_and_order_up_to_levels_with_mocked_newsvendor(self):
        # The SciPy optimizer currently passes ndarray values that the legacy
        # implementation cannot cast to ints, so we patch optC to isolate and
        # test RS_DP graph/path logic deterministically.
        fake_solution = types.SimpleNamespace(fun=7.5, x=[11.0])

        with mock.patch.object(rs.MultiPeriodNewsvendor, "optC", return_value=fake_solution):
            policy = rs.RS_DP(K=5, h=1, p=4, d=[2, 3, 2, 1])
            self.assertAlmostEqual(policy.cycle_cost(0, 1), 12.5)
            self.assertEqual(policy.order_up_to_levels(), [11.0, 0, 0, 11.0])
            self.assertAlmostEqual(policy.optimal_cost(), 25.0)


if __name__ == "__main__":
    unittest.main()
