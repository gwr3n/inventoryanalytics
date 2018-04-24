'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List

from docplex.mp.model import Model
import sys

sys.path.insert(0,'/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx')

class WagnerWhitin:
    '''
    A Wagner-Whitin problem.

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958
    '''

    def __init__(self, K: float, h: float, d: List[float]):
        '''
        Create an instance of a Wagner-Whitin problem.

        K: the fixed ordering cost;
        h: the per unit holding cost;
        d: the demand in each period.
        '''
        self.K, self.h, self.d = K, h, d
        self.model()

    def model(self):
        model = Model("Wagner Whitin")
        T = len(self.d)
        idx = [t for t in range(0,T)]
        self.Q = model.continuous_var_dict(idx, name="Q")
        I = model.continuous_var_dict(idx, name="I")
        delta = model.binary_var_dict(idx, name="delta")

        for t in range(0,T):
            model.add_constraint(model.sum(self.Q[k] - self.d[k] for k in range(0,t+1)) == I[t])
            model.add_constraint(model.if_then(delta[t] == 0, self.Q[t] == 0))

        model.minimize(model.sum(delta[t] * self.K + self.h * I[t] for t in range(0,T)))
        model.print_information()
        self.msol = model.solve()
        if self.msol:
            model.print_solution()
        else:
            print("Solve status: " + self.msol.get_solve_status() + "\n")
        
    def order_quantities(self) -> List[float]:
        '''
        Compute optimal Wagner-Whitin order quantities
        '''
        return [self.msol.get_var_value(self.Q[t]) for t in range(0,len(self.d))]
    
    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        '''
        return self.msol.get_objective_value()    

if __name__ == '__main__':
    instance = {"K": 30, "h": 1, "d":[10,20,30,40]}
    ww = WagnerWhitin(**instance)
    ww.model()