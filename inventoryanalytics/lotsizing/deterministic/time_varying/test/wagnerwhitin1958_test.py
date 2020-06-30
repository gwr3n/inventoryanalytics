'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.time_varying.wagnerwhitin1958 as ww

class TestWagnerWhitin(unittest.TestCase):

    def setUp(self):
        instance = {"K": 30, "h": 1, "d":[10,20,30,40]}
        self.ww_dp = ww.WagnerWhitinDP(**instance)
        self.ww_cplex = ww.WagnerWhitinCPLEX(**instance)

        instance_I0 = {"K": 30, "h": 1, "d":[10,20,30,40], "I0": 30}
        self.ww_dp_I0 = ww.WagnerWhitinDPI0(**instance_I0)
        self.ww_cplex_I0 = ww.WagnerWhitinCPLEX(**instance_I0)

    def tearDown(self):
        pass

    def test_cycle_cost(self):
        self.assertEqual(self.ww_dp.cycle_cost(0,2), 110) # 30 + 10*0 + 20*1 + 30*2 = 90
        self.assertEqual(self.ww_dp.cycle_cost(1,3), 140) # 30 + 20*0 + 30*1 + 40*2 = 120
    
    def test_optimal_cost(self):
        self.assertEqual(self.ww_dp.optimal_cost(), 110) # 30 + 10*0 + 20*1 + 30 + 30*0 + 40*1 = 120
        self.assertEqual(self.ww_cplex.optimal_cost(), 110) # 30 + 10*0 + 20*1 + 30 + 30*0 + 40*1 = 120

    def test_order_quantities(self):
        self.assertEqual(self.ww_dp.order_quantities(), [30,0,30,40])
        self.assertEqual(self.ww_cplex.order_quantities(), [30,0,30,40])

    def test_order_quantities_I0(self):
        self.assertEqual(self.ww_dp_I0.order_quantities(), [0,0,30,40])
        self.assertEqual(self.ww_cplex_I0.order_quantities(), [0,0,30,40])

if __name__ == '__main__':
    unittest.main()