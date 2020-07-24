'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from docplex.mp.model import Model
import sys

# sys.path.insert(0,'/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx')
sys.path.insert(0,'/Applications/CPLEX_Studio1210/cplex/python/3.7/x86-64_osx')

# http://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
# http://www-01.ibm.com/support/docview.wss?uid=swg27042869&aid=1

from typing import List

class WagnerWhitinPlannedBackorders:
    """
    A Wagner-Whitin problem with planned backorders.

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958
    """
    def __init__(self, K: float, v: float, h: float, p: float, d: List[float], I0: float):
        """
        Create an instance of a Wagner-Whitin problem.

        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the per unit ordering cost
            h {float} -- the per unit holding cost
            p {float} -- the per unit backorder cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """

        self.K, self.v, self.h, self.p, self.d, self.I0 = K, v, h, p, d, I0

class WagnerWhitinPlannedBackordersCPLEX(WagnerWhitinPlannedBackorders):
    """
    Solves the Wagner-Whitin problem as an MILP.
    """
    def model(self):
        """
        Model and solve the Wagner Whitin problem via CPLEX
        """

        model = Model("Wagner Whitin planned backorders")
        T = len(self.d)
        M = sum(self.d)
        idx = [t for t in range(0,T)]
        self.Q = model.continuous_var_dict(idx, name="Q")
        I = model.continuous_var_dict(idx, lb=-M, name="I")
        Ip = model.continuous_var_dict(idx, name="I^+")
        Im = model.continuous_var_dict(idx, name="I^-")
        delta = model.binary_var_dict(idx, name="delta")
        

        for t in range(0,T):
            model.add_constraint(self.Q[t] <= delta[t]*M)
            model.add_constraint(self.I0 + model.sum(self.Q[k] - self.d[k] for k in range(0,t+1)) == I[t])
            model.add_constraint(I[t] == Ip[t]-Im[t])
            model.add_constraint(self.Q[t] >= 0)
            model.add_constraint(Ip[t] >= 0)
            model.add_constraint(Im[t] >= 0)

        model.minimize(model.sum(delta[t] * self.K + self.Q[t] * self.v + self.h * Ip[t] + self.p * Im[t] for t in range(0,T)))
        model.print_information()
        self.msol = model.solve()
        if self.msol:
            model.print_solution()
        else:
            print("Solve status: " + self.msol.get_solve_status() + "\n")
        
    def order_quantities(self) -> List[float]:
        """
        Compute optimal Wagner-Whitin order quantities
        """
        return [self.msol.get_var_value(self.Q[t]) for t in range(0,len(self.d))]
    
    def optimal_cost(self) -> float:
        """
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        """
        return self.msol.get_objective_value()    

    @staticmethod
    def _test():
        instance = {"K": 40, "v": 1, "h": 1, "p": 2, "d":[10,20,30,40], "I0": 0}
        ww = WagnerWhitinPlannedBackordersCPLEX(**instance)
        ww.model()

if __name__ == '__main__':
    WagnerWhitinPlannedBackordersCPLEX._test()