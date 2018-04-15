'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.stochastic.nonstationary.sdp as sdp

class TestScarf1960(unittest.TestCase):

    def setUp(self):
        self.input = {"K": 100, "v": 0, "h":1, "p":10, "d":[20,40,60,40], 
                      "max_inv": 200, "q": 0.9999, "initial_order": True}

    def tearDown(self):
        pass

    @unittest.skip("Time consuming test")
    def test_scarf_1960(self):
        lot_sizing = sdp.StochasticLotSizing(**self.input)
        period = 0
        level = 0
        self.assertEqual(lot_sizing.f(level), 332.1194157143863)
        self.assertEqual(lot_sizing.q(period, level), 67)

    def test_state(self):
        state_1 = sdp.State(0,10)
        state_2 = sdp.State(0,20)
        state_3 = sdp.State(0,20)
        print(type(state_1))
        self.assertEqual(state_1 == state_2, False)
        self.assertEqual(state_2 == state_3, True)
        self.assertEqual(str(state_2), str(0)+" "+str(20))

    def test_constructor(self):
        lot_sizing = sdp.StochasticLotSizing(**self.input)
        self.assertEqual(lot_sizing.d, [20,40,60,40])

if __name__ == '__main__':
    unittest.main()