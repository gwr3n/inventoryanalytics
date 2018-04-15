'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.stochastic.nonstationary.policies as pol

class TestPolicies(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sS(self):
        sS = [[15, 67], [28, 49], [55, 109], [28, 49]]
        instance = {"inv": 0, "K": 100, "v": 0, "h": 1, "p": 10, "d": [20,40,60,40], "sS": sS, "replications": 1000}
        interval = pol.Policies.simulate_sS(**instance)
        self.assertEqual(interval[0] < 332.1194157143863, True)
        self.assertEqual(interval[1] > 332.1194157143863, True)