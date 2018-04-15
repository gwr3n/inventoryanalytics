'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.constant.eoq as eoq

class TestEOQ(unittest.TestCase):

    def setUp(self):
        instance = {"K": 3.2, "h": 0.24, "d": 2400, "p": 0.4}
        self.eoq = eoq.eoq(**instance)

    def tearDown(self):
        pass

    def test_eoq(self):
        self.assertAlmostEqual(self.eoq.compute_eoq(), 252.98, places=2)

    def test_eoq_cost(self):
        self.assertAlmostEqual(self.eoq.compute_eoq_cost(), 1020.72, places=2)

    def test_itr(self):
        self.assertAlmostEqual(self.eoq.compute_itr(), 18.97, places=2)

    def test_coverage(self):
        self.assertAlmostEqual(self.eoq.compute_coverage(), 1.26/12, places=2)