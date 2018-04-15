'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np
from scipy.optimize import minimize

class eoq:
    '''
    Ford W. Harris, How Many Parts to Make at Once, Factory, 
    The Magazine of Management, Volume 10, Number 2, February 1913, pp. 135–136, 152.

    Harris, Ford W. (1990). "How Many Parts to Make at Once". Operations Research. 
    38 (6): 947. doi:10.1287/opre.38.6.947.
    '''

    def __init__(self, K: float, h: float, d: float, p: float):
        """
        Constructs an instance of the Economic Order Quantity problem.
        
        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the proportional holding cost
            d {float} -- the demand per period
            p {float} -- the unit purchasing cost
        """

        self.K, self.h, self.d, self.p = K, h, d, p

    def compute_eoq(self) -> float:
        """
        Computes the Economic Order Quantity.
        
        Returns:
            float -- the Economic Order Quantity
        """

        x0 = 1 # start from a positive EOQ
        res = minimize(self.compute_eoq_cost, x0, method='nelder-mead', 
                       options={'xtol': 1e-8, 'disp': False})
        return res.x[0]

    def compute_eoq_cost(self, Q: float) -> float:
        """
        Computes the optimal cost per unit period.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        K, h, d, p = self.K, self.h, self.d, self.p
        return (K+Q*p)/(Q/d)+h*Q/2

    def compute_coverage(self) -> float:
        """
        Compute the number of periods of demand the 
        Economic Order Quantity will satisfy.
        
        Returns:
            float -- the number of periods of demand the 
                Economic Order Quantity will satisfy
        """

        d = self.d
        return self.compute_eoq()/d

    def compute_itr(self) -> float:
        """
        The Implied Turnover Ratio (ITR) represents the number of times 
        inventory is sold or used in a time period.
        
        Returns:
            float -- the Implied Turnover Ratio (ITR)
        """

        d = self.d
        return 2*d/self.compute_eoq()

    def reorder_point(self, lead_time: float) -> float:
        """
        Computes the reorder point for a given lead time.
        
        Arguments:
            lead_time {float} -- the given lead time
        
        Returns:
            float -- the reorder point
        """

        d = self.d
        return d*lead_time