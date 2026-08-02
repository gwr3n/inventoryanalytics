import unittest

import inventoryanalytics.lotsizing.stochastic.nonstationary.capacitated_sdp as cap_sdp


class TestCapacitatedSDP(unittest.TestCase):

    def test_state_and_demand_types(self):
        state_1 = cap_sdp.State(1, 3)
        state_2 = cap_sdp.State(1, 3)
        state_3 = cap_sdp.State(1, 4)

        self.assertEqual(state_1 == state_2, True)
        self.assertEqual(state_1 == state_3, False)
        self.assertEqual(str(state_1), "1 3")

        poisson_demand = cap_sdp.PoissonDemand([2], 0.99)
        normal_demand = cap_sdp.NormalDemand([10], 0.1, 0.99)
        pmf_demand = cap_sdp.PmfDemand([[[0, 1.0]]])

        self.assertGreater(len(poisson_demand.pmf[0]), 0)
        self.assertGreater(len(normal_demand.pmf[0]), 0)
        self.assertEqual(pmf_demand.pmf, [[[0, 1.0]]])

    def test_small_instance_value_action_and_policy_extraction(self):
        demand = cap_sdp.PmfDemand([[[0, 0.5], [1, 0.5]], [[0, 1.0]]])
        instance = {
            "K": 5,
            "B": 2,
            "v": 0,
            "h": 1,
            "p": 4,
            "w": 1,
            "d": demand,
            "min_inv": -2,
            "max_inv": 5,
            "initial_order": True,
        }

        lot_sizing = cap_sdp.StochasticLotSizing(**instance)

        self.assertEqual(lot_sizing.ag(cap_sdp.State(0, 0)), [0, 1, 2])
        self.assertEqual(lot_sizing.st(cap_sdp.State(0, 0), 2, 1), cap_sdp.State(1, 1))
        self.assertAlmostEqual(lot_sizing.f(0), 4.0)
        self.assertEqual(lot_sizing.q(0, 0), 0)
        self.assertEqual(lot_sizing.extract_skSk_policy(), [[[-1, 1]], []])

    def test_initial_order_flag_blocks_period_zero_orders(self):
        demand = cap_sdp.PmfDemand([[[0, 0.5], [1, 0.5]], [[0, 1.0]]])
        instance = {
            "K": 5,
            "B": 2,
            "v": 0,
            "h": 1,
            "p": 4,
            "w": 1,
            "d": demand,
            "min_inv": -2,
            "max_inv": 5,
            "initial_order": False,
        }

        lot_sizing = cap_sdp.StochasticLotSizing(**instance)

        self.assertEqual(lot_sizing.ag(cap_sdp.State(0, 0)), [0])
        self.assertEqual(lot_sizing.ag(cap_sdp.State(1, 0)), [0, 1, 2])
        self.assertAlmostEqual(lot_sizing.f(0), 4.0)
        self.assertEqual(lot_sizing.q(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
