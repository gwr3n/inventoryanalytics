'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import unittest
import inventoryanalytics.abc.abc_analysis as abc
import numpy as np

class TestEOQ(unittest.TestCase):

    def setUp(self):
        # prepare data
        d = abc.ABC()
        self.data = d.data

    def tearDown(self):
        pass

    def test_adu(self):
        item_no = np.matrix(self.data )[1:,0]                #item ids
        m = np.matrix(self.data )[1:,[1,2]].astype(np.float) #criteria scores
        unsorted_adu = abc.ABC.annual_dollar_usage(item_no, m)
        adu_result = [['s1','5840.64','A'],['s2','5670.0','A'],['s3','5037.12','A'],['s4','4769.56','A'],['s5','3478.7999999999997','A'],['s6','2936.56','A'],['s7','2820.0','A'],['s8','2640.0','A'],['s9','2423.52','A'],['s10','2407.5','A'],['s11','1075.2','B'],['s12','1043.5','B'],['s13','1038.0','B'],['s14','883.2','B'],['s15','854.4000000000001','B'],['S16','810.0','B'],['s17','703.6800000000001','B'],['s18','594.0','B'],['s19','570.0','B'],['s20','467.6','B'],['s21','463.59999999999997','B'],['s22','455.0','B'],['s23','432.5','B'],['s24','398.40000000000003','B'],['s25','370.5','C'],['s26','338.40000000000003','C'],['s27','336.12','C'],['s28','313.6','C'],['s29','268.68','C'],['s30','224.0','C'],['s31','216.0','C'],['s32','212.08','C'],['s33','197.92','C'],['s34','190.89000000000001','C'],['s35','181.8','C'],['s36','163.28','C'],['s37','150.0','C'],['s38','134.8','C'],['s39','119.2','C'],['s40','103.36','C'],['s41','79.2','C'],['s42','75.4','C'],['s43','59.78','C'],['s44','48.3','C'],['s45','34.4','C'],['s46','28.8','C'],['s47','25.380000000000003','C']]
        adu_result = [[k[0], float(k[1]), k[2]] for k in adu_result]
        self.assertEqual(unsorted_adu, adu_result)
        
