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

class TestEOQ_all_units_discounts(unittest.TestCase):

    def setUp(self):
        instance = {"K": 3.2, "h": 0.24, "d": 2400, "v": 0.4}
        self.eoq = eoq.eoq(**instance)

    def tearDown(self):
        pass

    def test_eoq(self):
        instance = {"K": 8, "h": 0.3, "d": 1300, "b": [400,800], "v": [0.75,0.72,0.68]}
        pb = eoq.eoq_all_units_discounts(**instance)
        Q = pb.compute_eoq()
        self.assertAlmostEqual(Q, 800, places=2)
        self.assertAlmostEqual(pb.cost(Q), 978.6, places=2)

class TestEOQ_incremental_discounts(unittest.TestCase):

    def setUp(self):
        instance = {"K": 3.2, "h": 0.24, "d": 2400, "v": 0.4}
        self.eoq = eoq.eoq(**instance)

    def tearDown(self):
        pass

    def test_eoq(self):
        instance = {"K": 8, "h": 0.3, "d": 1300, "b": [400,800], "v": [0.75,0.72,0.68]}
        pb = eoq.eoq_incremental_discounts(**instance)
        Q = pb.compute_eoq()
        self.assertAlmostEqual(Q, 304.05, places=2)
        self.assertAlmostEqual(pb.cost(Q), 1043.41, places=2)

class TestEOQ_planned_backorders(unittest.TestCase):

    def setUp(self):
        instance = {"K": 8, "h": 0.3*0.75, "d": 1300, "v": 75, "p": 5}
        self.eoq = eoq.eoq_planned_backorders(**instance)

    def tearDown(self):
        pass

    def test_eoq(self):
        K, h, d, p = self.eoq.K, self.eoq.h, self.eoq.d, self.eoq.p
        self.assertAlmostEqual(self.eoq.compute_eoq(), 
                               np.sqrt(2*K*d*(h+p)/(h*p)), 
                               places=2) # closed-form
        
class TestEPQ(unittest.TestCase):

    def setUp(self):
        instance = {"K": 8, "h": 0.3*0.75, "d": 1300, "v": 75, "p": 5}
        self.epq = eoq.epq(**instance)

    def tearDown(self):
        pass

    def test_epq(self):
        K, h, d, p = self.epq.K, self.epq.h, self.epq.d, self.epq.p
        rho = p/d
        self.assertAlmostEqual(self.epq.compute_epq(), 
                               np.sqrt(2*K*d/(h*(1-rho))), 
                               places=2) # closed-form