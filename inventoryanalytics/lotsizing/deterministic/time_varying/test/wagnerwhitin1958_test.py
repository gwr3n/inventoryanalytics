'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.time_varying.wagnerwhitin1958_sp as ww_sp
import inventoryanalytics.lotsizing.deterministic.time_varying.wagnerwhitin1958_cplex as ww_cplex

class TestWagnerWhitin(unittest.TestCase):

    def setUp(self):
        instance = {"K": 30, "h": 1, "d":[10,20,30,40]}
        self.ww_sp = ww_sp.WagnerWhitin(**instance)
        self.ww_cplex = ww_cplex.WagnerWhitin(**instance)

    def tearDown(self):
        pass

    def test_cycle_cost(self):
        self.assertEqual(self.ww_sp.cycle_cost(0,2), 110) # 30 + 10*0 + 20*1 + 30*2 = 90
        self.assertEqual(self.ww_sp.cycle_cost(1,3), 140) # 30 + 20*0 + 30*1 + 40*2 = 120
    
    def test_optimal_cost(self):
        self.assertEqual(self.ww_sp.optimal_cost(), 110) # 30 + 10*0 + 20*1 + 30 + 30*0 + 40*1 = 120
        self.assertEqual(self.ww_cplex.optimal_cost(), 110) # 30 + 10*0 + 20*1 + 30 + 30*0 + 40*1 = 120

    def test_order_quantities(self):
        self.assertEqual(self.ww_sp.order_quantities(), [30,0,30,40])
        self.assertEqual(self.ww_cplex.order_quantities(), [30,0,30,40])

if __name__ == '__main__':
    unittest.main()