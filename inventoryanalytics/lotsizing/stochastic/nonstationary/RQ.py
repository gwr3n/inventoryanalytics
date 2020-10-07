'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import sys
from docplex.mp.model import Model
# sys.path.insert(0,'/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx')
sys.path.insert(0,'/Applications/CPLEX_Studio1210/cplex/python/3.7/x86-64_osx')

# http://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
# http://www-01.ibm.com/support/docview.wss?uid=swg27042869&aid=1

import math, networkx as nx, itertools
from typing import List

class StochasticLotSizing:
    """
    A stochastic lot sizing problem.
    """
    def __init__(self, K: float, h: float, p: float, d: List[float], I0: float):
        """
        Create an instance of the stochastic lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            p {float} -- the per unit penalty cost
            d {List[float]} -- the mean demand in each period
            I0 {float} -- the initial inventory level
        """

        self.K, self.h, self.p, self.d, self.I0 = K, h, p, d, I0

class RQ_CPLEX(StochasticLotSizing):
    """
    Solves the RQ problem as an MILP.
    """

    def __init__(self, K: float, h: float, p: float, d: List[float], std_d: List[float], I0: float = 0):
        """
        Create an instance of a RQ problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """
        super().__init__(K, h, p, d, I0)
        self.std_demand = std_d
        self.W = 5 # 5 partitions, 6 piecewise segments 
        # constant linearisation parameters from [Rossi et al., 2014]
        self.prob = [0.1324110437406592, 0.23491250409192982, 0.26535290433482195, 0.23491250409192987, 0.13241104374065915]
        self.E = [-1.6180463502161044, -0.6914240068499904, 0, 0.6914240068499903, 1.6180463502161053]
        self.e = 0.022270929512393414
        self.model()

    def model(self):
        model = Model("RQ")
        T = len(self.d)
        idx = [t for t in range(0,T)]
        self.Q = model.continuous_var_dict(idx, name="Q")
        I = model.continuous_var_dict(idx, name="I") # E[I]
        Ip = model.continuous_var_dict(idx, name="Ip") # E[I^+]
        Im = model.continuous_var_dict(idx, name="Im") # E[I^-]
        delta = model.binary_var_dict(idx, name="delta")

        for t in range(T):
            model.add_constraint(model.if_then(delta[t] == 0, self.Q[t] == 0))
            model.add_constraint(self.I0 + model.sum(self.Q[k] - self.d[k] for k in range(t+1)) == I[t])
            model.add_constraint(I[t] == Ip[t] - Im[t])
            
            for n in range(self.W): # complementary first order loss function piecewise segments
                model.add_constraint(Ip[t] >= I[t] * sum(self.prob[k] for k in range(n+1)) - sum([self.prob[k]*self.E[k] for k in range(n+1)]) * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])) + self.e * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])))
            model.add_constraint(Ip[t] >= self.e * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])))
            
            for n in range(self.W): # first order loss function piecewise segments
                model.add_constraint(Im[t] >= -I[t] + I[t] * sum(self.prob[k] for k in range(n+1)) - sum([self.prob[k]*self.E[k] for k in range(n+1)]) * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])) + self.e * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])))
            model.add_constraint(Im[t] >= -I[t] + self.e * math.sqrt(sum([self.std_demand[k]**2 for k in range(t+1)])))

            model.add_constraint(Ip[t] >= 0)
            model.add_constraint(Im[t] >= 0)

        model.minimize(model.sum(delta[t] * self.K + self.h * Ip[t] + self.p * Im[t] for t in range(T)))
        model.print_information()
        self.msol = model.solve()
        if self.msol:
            model.print_solution()
        else:
            print("Solve status: " + self.msol.get_solve_status() + "\n")
        
    def order_quantities(self) -> List[float]:
        '''
        Compute optimal RQ order quantities
        '''
        return [self.msol.get_var_value(self.Q[t]) for t in range(0,len(self.d))]
    
    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution
        '''
        return self.msol.get_objective_value()    

    @staticmethod
    def _test():
        instance = {"K": 300, "h": 1, "p": 20, "d":[100,100,100,100,100,100,100,100], "std_d" : [10,10,10,10,10,10,10,10], "I0": 0}
        ww = RQ_CPLEX(**instance)
        print(ww.order_quantities())
        print(ww.optimal_cost())

if __name__ == '__main__':
    RQ_CPLEX._test()