'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
import networkx as nx
import scipy.integrate as integrate
from itertools import accumulate
from scipy.stats import poisson
from scipy.optimize import minimize

class MultiPeriodNewsvendor:
    def __init__(self, mean, h, p):
        self.mean, self.o, self.u = mean, h, p    

    def cfolf(self, Q, d): # complementary first order loss function
        return integrate.quad(lambda x: poisson.cdf(x, d), 0, Q)[0]

    def folf(self,Q): # first order loss function
        return self.cfolf(Q)-self.u*(Q - self.mean)

    def _C(self, Q): # C(Q)
        return sum([(self.o+self.u)*self.cfolf(Q, d)-self.u*(Q - d) for d in accumulate(self.mean)])

    def optC(self): # min C(Q)
        return minimize(self._C, 0, method='Nelder-Mead')

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
            d {List[float]} -- the poisson demand rate in each period
            I0 {float} -- the initial inventory level
        """

        self.K, self.h, self.p, self.d, self.I0 = K, h, p, d, I0
        
class RS_DP(StochasticLotSizing):
    """
    Implements the traditional shortest path algorithm with stochastic cycle costs.

    James H. Bookbinder and Jin-Yan Tan. Strategies for the probabilistic lot-sizing 
    problem with service-level constraints. Management Science, 34(9):1096-1108, 1988.
    """

    def __init__(self, K: float, h: float, p: float, d: List[float]):
        """
        Create an instance of the stochastic lot sizing problem.
        Initial inventory level assumed to be 0.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            p {float} -- the per unit penalty cost
            d {List[float]} -- the demand in each period
        """
        super().__init__(K, h, p, d, 0)

        self.graph = nx.DiGraph()
        for i in range(0, len(self.d)):
            for j in range(i+1, len(self.d)):
                self.graph.add_edge(i, j, weight=self.cycle_cost(i, j-1))

    def cycle_cost(self, i: int, j: int) -> float:
        '''
        Compute the expected total cost of a replenishment cycle covering periods i,...,j
        when initial inventory is zero
        '''
        if i>j: raise Exception('i>j')

        return self.K + MultiPeriodNewsvendor(self.d[i:j+1],self.h,self.p).optC().fun

    def optimal_cost(self) -> float:
        '''
        Approximates the cost of an optimal solution to the stochastic lot sizing problem
        '''
        T, cost, g = len(self.d), 0, self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        for t in range(1,len(path)):
            cost += self.cycle_cost(path[t-1],path[t]-1)
            print("c("+str(path[t-1])+","+str(path[t]-1)+") = "+str(self.cycle_cost(path[t-1],path[t]-1)))
        return cost
    
    def order_up_to_levels(self) -> List[float]:
        '''
        Compute optimal order-up-to-levels
        '''
        T, g = len(self.d), self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        qty = [0 for k in range(0,T)]
        for t in range(1,len(path)):
            qty[path[t-1]] = MultiPeriodNewsvendor(self.d[path[t-1]:path[t]],self.h,self.p).optC().x[0]
        return qty

    @staticmethod
    def _test():
        instance = {"K": 100, "h": 1, "p": 10, "d":[20,40,60,40]}
        ww = RS_DP(**instance)
        print("Cost of an optimal plan: ", ww.optimal_cost())
        print("Optimal order-up-to-levels: ", ww.order_up_to_levels())

if __name__ == '__main__':
    RS_DP._test()