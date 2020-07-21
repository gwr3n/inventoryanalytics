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
import networkx as nx
import itertools

class WagnerWhitin:
    """
    A Wagner-Whitin problem.

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958
    """
    def __init__(self, K: float, h: float, d: List[float], I0: float):
        """
        Create an instance of a Wagner-Whitin problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """

        self.K, self.h, self.d, self.I0 = K, h, d, I0

class WagnerWhitinCPLEX(WagnerWhitin):
    """
    Solves the Wagner-Whitin problem as an MILP.
    """

    def __init__(self, K: float, h: float, d: List[float], I0: float = 0):
        """
        Create an instance of a Wagner-Whitin problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """
        super().__init__(K, h, d, I0)
        self.model()

    def model(self):
        """
        Model and solve the Wagner Whitin problem via CPLEX
        """

        model = Model("Wagner Whitin")
        T = len(self.d)
        idx = [t for t in range(0,T)]
        self.Q = model.continuous_var_dict(idx, name="Q")
        I = model.continuous_var_dict(idx, name="I")
        delta = model.binary_var_dict(idx, name="delta")

        for t in range(0,T):
            model.add_constraint(self.I0 + model.sum(self.Q[k] - self.d[k] for k in range(0,t+1)) == I[t])
            model.add_constraint(model.if_then(delta[t] == 0, self.Q[t] == 0))

        model.minimize(model.sum(delta[t] * self.K + self.h * I[t] for t in range(0,T)))
        model.print_information()
        self.msol = model.solve()
        if self.msol:
            model.print_solution()
            print(model.solution.get_value("Q_2"))
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

    @staticmethod
    def _test():
        instance = {"K": 30, "h": 1, "d":[10,20,30,40], "I0": 0}
        ww = WagnerWhitinCPLEX(**instance)
        
class WagnerWhitinDP(WagnerWhitin):
    """
    Implements the traditional Wagner-Whitin shortest path algorithm.
    """

    def __init__(self, K: float, h: float, d: List[float], I0: float = 0):
        """
        Create an instance of a Wagner-Whitin problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
        """
        super().__init__(K, h, d, I0)

        if I0 > 0:
            self.cycle_cost = self.cycle_cost_I0

        self.graph = nx.DiGraph()
        for i in range(0, len(self.d)):
            for j in range(i+1, len(self.d)):
                self.graph.add_edge(i, j, weight=self.cycle_cost(i, j-1))

    def cycle_cost(self, i: int, j: int) -> float:
        '''
        Compute the cost of a replenishment cycle covering periods i,...,j
        when initial inventory is zero
        '''
        if i>j: raise Exception('i>j')

        return self.K + self.h * sum([(k-i)*self.d[k] for k in range(i,j+1)]) 

    def cycle_cost_I0(self, i: int, j: int) -> float:
        '''
        Compute the cost of a replenishment cycle covering periods i,...,j
        when initial inventory is nonzero
        '''
        if i>j: raise Exception('i>j')
                          
        if i == 0 and sum(self.d[0:j+1]) <= self.I0: 
            return self.h * sum([(k-i)*self.d[k] for k in range(i,j+1)]) + \
                   self.h * (j+1) * (self.I0-sum(self.d[0:j+1])) # cost no order
        elif i > 0 and sum(self.d[0:j+1]) <= self.I0:
            return sys.maxsize
        else: 
            return self.K + self.h * \
                   sum([(k-i)*self.d[k] for k in range(i,j+1)]) # cost with order

    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        '''
        T, cost, g = len(self.d), 0, self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        for t in range(1,len(path)):
            cost += self.cycle_cost(path[t-1],path[t]-1)
            print("c("+str(path[t-1])+","+str(path[t]-1)+") = "+str(self.cycle_cost(path[t-1],path[t]-1)))
        return cost
    
    def order_quantities(self) -> List[float]:
        '''
        Compute optimal Wagner-Whitin order quantities
        '''
        T, g = len(self.d), self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        qty = [0 for k in range(0,T)]
        for t in range(1,len(path)):
            qty[path[t-1]] = sum([self.d[k] for k in range(path[t-1],path[t])]) if sum(self.d[0:path[t-1]+1]) > self.I0 else 0
        return qty

    @staticmethod
    def _test():
        instance = {"K": 30, "h": 1, "d":[10,20,30,40], "I0": 30}
        ww = WagnerWhitinDP(**instance)
        print("Cost of an optimal plan: ", ww.optimal_cost())
        print("Optimal order quantities: ", ww.order_quantities())

if __name__ == '__main__':
    # WagnerWhitinCPLEX._test()
    WagnerWhitinDP._test()