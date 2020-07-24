'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.constant.els as els

class TestELS(unittest.TestCase):

    def setUp(self):
        instance = {"n": 3, "p": [400,400,500], "d": [50,50,60], "h": [20,20,30], "s":[0.1,0.1,0.1], "K": [2000,2500,800]}
        self.els = els.els(**instance)

    def tearDown(self):
        pass

    def test_els(self):
        self.assertAlmostEqual(
            self.els.compute_els(), 
            self.els._compute_els_closed_form(), 
            places=2)