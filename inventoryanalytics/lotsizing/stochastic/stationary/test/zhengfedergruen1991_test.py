'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.lotsizing.stochastic.stationary.zhengfedergruen1991 as zf

class TestWagnerWhitin(unittest.TestCase):

    def setUp(self):
        self.data = {'mu': 10, 'K': 64, 'h': 1., 'b': 9.}

    def test_instance_1(self):
        self.data['mu'] = 10
        pb = zf.ZhengFedergruen(**self.data)
        pb.findOptimalPolicy()
        self.assertEqual(pb.c(6,40), 35.02155527232042)
        
    def test_instance_2(self):
        self.data['mu'] = 20
        pb = zf.ZhengFedergruen(**self.data)
        pb.findOptimalPolicy()
        self.assertEqual(pb.c(14,62), 49.173035744939355)

    def test_instance_3(self):
        self.data['mu'] = 64
        pb = zf.ZhengFedergruen(**self.data)
        pb.findOptimalPolicy()
        self.assertEqual(pb.c(55,74), 78.40232070917746)