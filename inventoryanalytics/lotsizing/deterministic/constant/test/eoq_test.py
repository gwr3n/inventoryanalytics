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
        instance = {"K": 3.2, "h": 0.24, "d": 2400, "v": 0.4}
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
        K, h, d, v = self.eoq.K, self.eoq.h, self.eoq.d, self.eoq.v
        self.assertAlmostEqual(self.eoq.cost(Q), 
                               np.sqrt(2*K*h*d)+v*d, places=2) # closed-form
    
    def test_relevant_cost(self):
        Q = self.eoq.compute_eoq()
        K, h, d = self.eoq.K, self.eoq.h, self.eoq.d
        self.assertAlmostEqual(self.eoq.relevant_cost(Q), 
                               np.sqrt(2*K*h*d), places=2) # closed-form

    def test_itr(self):
        self.assertAlmostEqual(self.eoq.itr(), 18.97, places=2)

    def test_sensitivity_to_Q(self):
        Q = 30
        Qopt = self.eoq.compute_eoq()
        d, v = self.eoq.d, self.eoq.v
        self.assertAlmostEquals(self.eoq.sensitivity_to_Q(Q), (self.eoq.cost(Q)-d*v)/(self.eoq.cost(Qopt)-d*v), places=2)

    def test_reorder_point(self):
        L = 1/12
        self.assertAlmostEquals(self.eoq.reorder_point(L), 200, places=2)

    def test_coverage(self):
        self.assertAlmostEqual(self.eoq.coverage(), 1.26/12, places=2)