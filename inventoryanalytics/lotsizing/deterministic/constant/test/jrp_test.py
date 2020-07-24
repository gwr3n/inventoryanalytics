'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.constant.jrp as jrp

class TestELS(unittest.TestCase):

    def setUp(self):
        instance = {"n": 5, "beta": 52, "h":[1,1,1,1,1], "d":[2,2,2,2,2], "K":[1,2,4,6,16], "K0": 5}
        self.jrp = jrp.jrp(**instance)

    def tearDown(self):
        pass

    def test_els(self):
        self.assertAlmostEqual(
            self.jrp.solve(), 
            25.3317, 
            places=2)