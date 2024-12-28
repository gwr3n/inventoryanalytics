'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
import time
import math
import functools 

import networkx as nx
import scipy.stats as stats

class memoize(object): 
    """
    Memoization utility
    """

    def __init__(self, func): 
        self.func = func 
        self.memoized = {} 
        self.method_cache = {} 

    def __call__(self, *args): 
        return self.cache_get(self.memoized, args, 
            lambda: self.func(*args)) 

    def __get__(self, obj, objtype): 
        return self.cache_get(self.method_cache, obj, 
            lambda: self.__class__(functools.partial(self.func, obj))) 

    def cache_get(self, cache, key, func): 
        try: 
            return cache[key] 
        except KeyError: 
            cache[key] = func() 
            return cache[key] 
    
    def reset(self):
        self.memoized = {} 
        self.method_cache = {} 

def minimize(function, xmin, xmax):
    while xmin < xmax:
        xmid = (xmin + xmax) // 2
        if function(xmid) < function(xmid + 1):
            xmax = xmid
        else:
            xmin = xmid + 1
    return {"min_value": function(xmin), "argmin": xmin}

@staticmethod
@memoize
def cfolf(lmbda, Q):
    """
    Compute the complementary first order loss function for a Poisson random variable with rate lambda and initial inventory Q.
    
    Parameters:
    lmbda (float): The rate parameter of the Poisson distribution.
    Q (int): The initial inventory level.
    
    Returns:
    float: The value of the complementary first order loss function.
    """
    loss = sum(stats.poisson.cdf(k, lmbda) for k in range(int(Q)))
    return loss

@staticmethod
@memoize
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
    A non-stationary stochastic lot sizing problem under penalty costs, see e.g.

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
    Implements the traditional Wagner-Whitin shortest path algorithm 

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958

    in a stochastic setting under penalty costs, as discussed in 

    S. A. Tarim. Dynamic Lotsizing Models for Stochastic Demand in Single and
    Multi-Echelon Inventory Systems. PhD thesis, Lancaster University, 1996.
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

        self.graph = nx.DiGraph()
        start_time = time.time()
        for i in range(0, len(self.d)):
            for j in range(i+1, len(self.d) + 1):
                self.graph.add_edge(i, j, weight=self.cycle_cost(i, j-1))
        self.precompute_time = time.time() - start_time

        # Print the connection matrix
        # adj_matrix = nx.adjacency_matrix(self.graph).todense()
        # print("Connection Matrix ("+str(round(self.precompute_time, 2))+"):")
        # print(adj_matrix) 

    @memoize
    def cycle_cost(self, i: int, j: int) -> float:
        '''
        Compute the expected total cost of a replenishment cycle covering periods i,...,j
        '''
        if i>j: raise Exception('i>j')

        expTotalCost = lambda Q : sum([self.h*cfolf(sum([self.d[j] for j in range(i,k+1)]),Q) + self.p*folf(sum([self.d[j] for j in range(i,k+1)]),Q) for k in range(i,j+1)])

        if i == 0 and not math.isnan(self.I0): 
            return expTotalCost(self.I0) # cost no order
        else: 
            return self.K + minimize(expTotalCost, 0, int(sum(self.d)+3*math.sqrt(sum(self.d))))["min_value"]
    
    @memoize
    def cycle_S(self, i: int, j: int) -> float:
        '''
        Compute the order up to level of a replenishment cycle covering periods i,...,j
        '''
        if i>j: raise Exception('i>j')

        expTotalCost = lambda Q : self.K + sum([self.h*cfolf(sum([self.d[j] for j in range(i,k+1)]),Q) + self.p*folf(sum([self.d[j] for j in range(i,k+1)]),Q) for k in range(i,j+1)])

        if i == 0 and not math.isnan(self.I0): 
            return self.I0
        else:
            return minimize(expTotalCost, 0, int(sum(self.d)+3*math.sqrt(sum(self.d))))["argmin"]

    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        '''
        T, cost, g = len(self.d), 0, self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        for t in range(1,len(path)):
            cost += self.cycle_cost(path[t-1],path[t]-1)
            # print("c("+str(path[t-1])+","+str(path[t]-1)+") = "+str(self.cycle_cost(path[t-1],path[t]-1)))
        return cost
    
    def order_up_to_level(self) -> List[float]:
        '''
        Compute optimal order up to levels
        '''
        T, g = len(self.d), self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        qty = [0 for k in range(0,T)]
        for t in range(1,len(path)):
            qty[path[t-1]] = self.cycle_S(path[t-1],path[t]-1)
        return qty
    
    @staticmethod
    def reorder_points(instance) -> List[float]:
        '''
        Compute the reorder points
        '''
        ins = {"K": instance["K"], "h": instance["h"], "p": instance["p"], "d":instance["d"].copy()}
        d = instance["d"].copy()
        s = []
        S = []
        while len(d) > 0:
            ww = RS_DP(**ins)
            optCost = ww.optimal_cost()
            S0 = ww.order_up_to_level()[0]
            S.append(S0)
            cost = optCost
            I0 = S0
            while cost - optCost < ins["K"]:
                I0 -= 1
                instance_I0 = {"K": ins["K"], "h": ins["h"], "p": ins["p"], "d":d, "I0":I0}
                ww = RS_DP(**instance_I0)
                cost = ww.optimal_cost()
                # print('.', end='', flush=True)
            s.append(I0)
            d.pop(0)
            ins["d"] = d
        return {"S": S, "s": s}
    
    def simulate_sS(self, s, S, initial_inventory):
        '''
        Simulate an (s,S) policy
        '''
        ETC = 0
        replications = 10000
        realisations = [stats.poisson.rvs(self.d[t], size=replications) for t in range(len(self.d))]
        for r in range(replications):
            inv = initial_inventory
            cost = 0
            for t in range(len(self.d)):
                if inv <= s[t]:
                    a = S[t] - inv
                    inv = S[t] - realisations[t][r]
                    cost += self.K 
                else:
                    inv -= realisations[t][r]
                cost += self.h*max(inv,0)+self.p*max(-inv,0)
            ETC += cost
        return ETC/replications

    @staticmethod
    def _test():
        instance = {"K": 100, "h": 1, "p": 10, "d":[20,40,60,40]}
        ww = RS_DP(**instance)
        start_time = time.time()
        optCost = ww.optimal_cost()
        end_time = time.time() - start_time
        res = ww.reorder_points(instance)
        print({'optCost': optCost, 'solTime': round(end_time + ww.precompute_time, 2),'S': res["S"], 's': res["s"]})
        print("Simulated cost: " + str(round(ww.simulate_sS(res["s"],res["S"],0),2)))

if __name__ == '__main__':
    RS_DP._test()