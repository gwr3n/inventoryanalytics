'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from docplex.cp.model import CpoModel
from sys import stdout
from typing import List

class jrp:
    def __init__(self, n:int, beta:int, h:List[float], d:List[float], K:List[float], K0:float):
        """An instance for the Joint Replenishment Problem

        Args:
            n (int): the number of items
            beta (int): the base planning period
            h (List[float]): holding cost rate for item
            d (List[float]): demand rate for item
            K (List[float]): fixed minor setup cost for item
            K0 (float): fixed major setup cost.
        """
        self.n, self.beta, self.h, self.d, self.K = n, beta, h, d, K
        self.H = [0] + [0.5 * h[i] * d[i] for i in range(0,n)]
        self.K = [K0] + K
        self.U = 30

    def solve(self):
        mdl = CpoModel()

        power = [2**i for i in range(0,self.U+1)]
        M = mdl.integer_var_list(self.n+1, 0, self.U, "M")
        T = mdl.integer_var_list(self.n+1, 0, power[self.U], "T")

        mdl.add(mdl.element(power, M[i]) == T[i] for i in range(0,self.n+1))
        mdl.add(T[i] >= T[0] for i in range(0,self.n+1))

        mdl.minimize(mdl.sum(self.H[i]*T[i]/self.beta+self.K[i]/(T[i]/self.beta) for i in range(0,self.n+1)))

        # Solve model
        print("Solving model....")
        msol = mdl.solve(TimeLimit=10,
                         agent='local',
                         execfile='/Applications/CPLEX_Studio1210/cpoptimizer/bin/x86-64_osx/cpoptimizer')

        # Print solution
        if msol:
            stdout.write("Total cost : "+str(msol.get_objective_values()[0]) + "\n")
            stdout.write("T: [")
            for t in T:
                stdout.write(" {}".format(msol[t]))
            stdout.write("]\n")
            stdout.write("M: [")
            for m in M:
                stdout.write(" {}".format(msol[m]))
            stdout.write("]\n")
            return msol.get_objective_values()[0]
        else:
            stdout.write("Solve status: {}\n".format(msol.get_solve_status()))
            return None

if __name__ == '__main__':
    instance = {"n": 5, "beta": 52, "h":[1,1,1,1,1], "d":[2,2,2,2,2], "K":[1,2,4,6,16], "K0": 5}
    jrp = jrp(**instance)
    jrp.solve()