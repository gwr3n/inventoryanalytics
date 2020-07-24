'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.time_varying.planned_backorders as ww

class TestWagnerWhitinPlannedBackorders(unittest.TestCase):

    def setUp(self):
        instance = {"K": 40, "v": 1, "h": 1, "p": 2, "d":[10,20,30,40], "I0": 0}
        self.ww_cplex = ww.WagnerWhitinPlannedBackordersCPLEX(**instance)
        self.ww_cplex.model()

    def tearDown(self):
        pass

    def test_optimal_cost(self):
        self.assertEqual(self.ww_cplex.optimal_cost(), 230) 
    
    def test_order_quantities(self):
        for e in zip(self.ww_cplex.order_quantities(), [0,60,0,40]):
            self.assertAlmostEqual(e[0],e[1])

if __name__ == '__main__':
    unittest.main()