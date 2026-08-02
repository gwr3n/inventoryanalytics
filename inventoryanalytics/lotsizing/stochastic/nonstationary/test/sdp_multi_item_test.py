import unittest

import inventoryanalytics.lotsizing.stochastic.nonstationary.sdp_multi_item as sdp_mi


class TestMultiItemScarf1960(unittest.TestCase):

    def test_state_behaviour(self):
        state_1 = sdp_mi.State(0, (1, 2))
        state_2 = sdp_mi.State(0, (2, 1))
        state_3 = sdp_mi.State(0, (2, 1))

        self.assertEqual(state_1 == state_2, False)
        self.assertEqual(state_2 == state_3, True)
        self.assertEqual(str(state_1), "0 (1, 2)")

    def test_small_instance_optimal_value_and_action(self):
        instance = {"K": 10, "v": 0, "h": 1, "p": 5, "d": [3],
                    "max_inv": 5, "q": 0.99, "initial_order": True}
        lot_sizing = sdp_mi.MultiItemStochasticLotSizing(**instance)

        self.assertAlmostEqual(lot_sizing.f((0, 0)), 15.308036756668638)
        self.assertEqual(lot_sizing.q(0, (0, 0)), (4, 4))

    def test_initial_order_flag_restricts_first_period_actions(self):
        instance = {"K": 10, "v": 0, "h": 1, "p": 5, "d": [3, 6],
                    "max_inv": 5, "q": 0.99, "initial_order": False}
        lot_sizing = sdp_mi.MultiItemStochasticLotSizing(**instance)

        self.assertEqual(lot_sizing.ag(sdp_mi.State(0, (0, 0))), [(0, 0)])
        self.assertEqual(len(lot_sizing.ag(sdp_mi.State(1, (0, 0)))), 25)
        self.assertAlmostEqual(lot_sizing.f((0, 0)), 59.948248711329136)
        self.assertEqual(lot_sizing.q(0, (0, 0)), (0, 0))


if __name__ == "__main__":
    unittest.main()
