'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.deterministic.constant.eoq as eoq
import numpy as np

class TestEOQ(unittest.TestCase):

    def setUp(self):
        instance = {"K": 3.2, "h": 0.24, "d": 2400, "p": 0.4}
        self.eoq = eoq.eoq(**instance)

    def tearDown(self):
        pass

    def test_eoq(self):
        K, h, d = self.eoq.K, self.eoq.h, self.eoq.d
        self.assertAlmostEqual(self.eoq.compute_eoq(), 
                               np.sqrt(2*d*K/h), places=2) # closed-form
        self.assertAlmostEqual(self.eoq.compute_eoq(), 252.98, places=2)

    def test_cost(self):
        Q = self.eoq.compute_eoq()
        self.assertAlmostEqual(1020.72, self.eoq.cost(Q), places=2)
        K, h, d, p = self.eoq.K, self.eoq.h, self.eoq.d, self.eoq.p
        self.assertAlmostEqual(self.eoq.cost(Q), 
                               np.sqrt(2*K*h*d)+p*d, places=2) # closed-form

    def test_itr(self):
        self.assertAlmostEqual(self.eoq.itr(), 18.97, places=2)

    def test_coverage(self):
        self.assertAlmostEqual(self.eoq.coverage(), 1.26/12, places=2)