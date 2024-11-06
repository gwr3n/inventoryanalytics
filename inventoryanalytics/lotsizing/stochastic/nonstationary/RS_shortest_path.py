'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import sys
from typing import List
import time
import math

import networkx as nx
import scipy.stats as stats

@staticmethod
def cfolf(lmbda, Q):
    """
    Compute the complementary first order loss function for a Poisson random variable with rate lambda and initial inventory Q.
    
    Parameters:
    lmbda (float): The rate parameter of the Poisson distribution.
    Q (int): The initial inventory level.
    
    Returns:
    float: The value of the complementary first order loss function.
    """
    loss = 0
    for k in range(int(Q)):
        #loss += (Q - k) * (math.exp(-lmbda) * (lmbda ** k) / math.factorial(k))
        loss += stats.poisson.cdf(k, lmbda)
    return loss

@staticmethod
def folf(lmbda, Q):
    """
    Compute the first order loss function for a Poisson random variable with rate lambda and initial inventory Q.
    
    Parameters:
    lmbda (float): The rate parameter of the Poisson distribution.
    Q (int): The initial inventory level.
    
    Returns:
    float: The value of the first order loss function.
    """
    return cfolf(lmbda, Q) - (Q - lmbda)

class StochasticLotSizingProblem:
    """
    A non-stationary stochastic lot sizing problem.

    R. Rossi, O. A. Kilic, S. A. Tarim, 
    "Piecewise linear approximations for the static-dynamic uncertainty strategy in stochastic lot-sizing", 
    OMEGA - the International Journal of Management Science, Elsevier, Vol. 50:126-140, 2015
    """
    def __init__(self, K: float, h: float, p: float, d: List[float], I0: float):
        """
        Create an instance of a non-stationary stochastic lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            p {float} -- the per unit shortage cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """

        self.K, self.h, self.p, self.d, self.I0 = K, h, p, d, I0
        
class RS_DP(StochasticLotSizingProblem):
    """
    Implements the traditional Wagner-Whitin shortest path algorithm.

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958
    """

    def __init__(self, K: float, h: float, p: float, d: List[float], I0: float = float('nan')):
        """
        Create an instance of a non-stationary stochastic lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the per unit holding cost
            p {float} -- the per unit shortage cost
            d {List[float]} -- the demand in each period
        """
        super().__init__(K, h, p, d, I0)

        if (not math.isnan(I0)):
            self.cycle_cost = self.cycle_cost_I0

        self.graph = nx.DiGraph()
        for i in range(0, len(self.d)):
            for j in range(i+1, len(self.d)):
                self.graph.add_edge(i, j, weight=self.cycle_cost(i, j-1))
        
        # Print the connection matrix
        adj_matrix = nx.adjacency_matrix(self.graph).todense()
        print("Connection Matrix:")
        print(adj_matrix)

    def cycle_cost(self, i: int, j: int) -> float:
        '''
        Compute the expected total cost of a replenishment cycle covering periods i,...,j
        '''
        if i>j: raise Exception('i>j')

        expTotalCost = lambda Q : self.K + sum([self.h*cfolf(sum([self.d[j] for j in range(i,k+1)]),Q) + self.p*folf(sum([self.d[j] for j in range(i,k+1)]),Q) for k in range(i,j+1)])

        return min([expTotalCost(k) for k in range(int(sum(self.d)+3*math.sqrt(sum(self.d))))])
    
    def cycle_S(self, i: int, j: int) -> float:
        '''
        Compute the order up to level of a replenishment cycle covering periods i,...,j
        '''
        if i>j: raise Exception('i>j')

        expTotalCost = lambda Q : self.K + sum([self.h*cfolf(sum([self.d[j] for j in range(i,k+1)]),Q) + self.p*folf(sum([self.d[j] for j in range(i,k+1)]),Q) for k in range(i,j+1)])

        values = [expTotalCost(k) for k in range(int(sum(self.d)+3*math.sqrt(sum(self.d))))]
        return values.index(min(values))

    def cycle_cost_I0(self, i: int, j: int) -> float:
        '''
        Compute the cost of a replenishment cycle covering periods i,...,j
        when initial inventory is nonzero
        '''
        if i>j: raise Exception('i>j')

        expTotalCost = lambda Q : sum([self.h*cfolf(sum([self.d[j] for j in range(i,k+1)]),Q) + self.p*folf(sum([self.d[j] for j in range(i,k+1)]),Q) for k in range(i,j+1)])

        if i == 0: 
             return expTotalCost(self.I0) # cost no order
        else: 
            return self.K + min([expTotalCost(k) for k in range(int(sum(self.d)+3*math.sqrt(sum(self.d))))]) # cost with order

    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        '''
        T, cost, g = len(self.d), 0, self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        for t in range(1,len(path)):
            cost += self.cycle_cost(path[t-1],path[t]-1)
            #print("c("+str(path[t-1])+","+str(path[t]-1)+") = "+str(self.cycle_cost(path[t-1],path[t]-1)))
        return cost
    
    def order_up_to_level(self) -> List[float]:
        '''
        Compute optimal Wagner-Whitin order quantities
        '''
        T, g = len(self.d), self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        qty = [0 for k in range(0,T)]
        for t in range(1,len(path)):
            qty[path[t-1]] = self.cycle_S(path[t-1],path[t]-1)
        return qty

    @staticmethod
    def _test():
        instance = {"K": 50, "h": 1, "p": 10, "d":[30,40,30,40,30]}
        ww = RS_DP(**instance)
        start_time = time.time()
        optCost = ww.optimal_cost()
        end_time = time.time() - start_time
        print({'optCost': optCost, 'solTime': round(end_time, 2),'S': ww.order_up_to_level()})

if __name__ == '__main__':
    RS_DP._test()