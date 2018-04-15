
'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
import numpy as np
import numpy.random as rnd
import scipy.stats as sp

class Policies:

    @staticmethod
    def simulate_sS(inv: float, K: float, v: float, h: float, 
                    p: float, d: List[float], sS: List[float], 
                    replications: int) -> List[float]:
        """
        Simulate the (s,S) inventory policy (Poisson demand).
        
        Arguments:
            inv {float} -- the initial inventory level
            K {float} -- the first ordering cost
            v {float} -- the proportional unit ordering cost
            h {float} -- the proportional holding cost
            p {float} -- the proportional penalty cost
            d {List[float]} -- the demand rate in each period
            sS {List[float]} -- the s,S thresholds in each period, 
               a list formatted as follows: [[s_1,S_1],...,[s_n,S_n]]
            replications {int} -- the simulation runs
        
        Returns:
            List[float] -- a confidence interval [lb,ub] for the 
               simulated expected total cost
        """

        N = replications
        rd = Policies._simulate_random_demand(d, N)
        tally = [Policies._simulate_sS_one_run(inv, K, v, h, p, rd[r], sS) 
                 for r in range(0, len(rd))]
        return sp.t.interval(0.95, len(tally)-1, 
                             loc=np.mean(tally), 
                             scale=sp.sem(tally))

    @staticmethod
    def _simulate_sS_one_run(inv: float, K: float, v: float, h: float, 
                             p: float, d: float, sS: List[float]) -> float:
        i = inv
        c = lambda t, i: K + v*(sS[t][1]-i) if i < sS[t][0] else 0
        L = lambda t, i: h*max(i,0) + p*max(-i,0)
        cost = 0
        for t in range(0, len(d)):
            cost += c(t, i)
            i = sS[t][1] - d[t] if i < sS[t][0] else i - d[t]
            cost += L(t, i)
        return cost

    @staticmethod
    def _simulate_random_demand(d: List[float], replications: int) -> List[float]:
        N = replications
        realisations = rnd.rand(N, len(d))
        demand = [[sp.poisson.ppf(realisations[r,t], d[t]) 
                   for t in range(0,len(d))] 
                   for r in range(0,N)]
        return demand

    @staticmethod
    def simulate_sample_policy():
        '''
        Optimal policy cost: 332.1194157143863
        Optimal order quantity: 67
        [[15, 67], [28, 49], [55, 109], [28, 49]]
        '''
        sS = [[15, 67], [28, 49], [55, 109], [28, 49]]
        instance = {"inv": 0, "K": 100, "v": 0, "h": 1, 
                    "p": 10, "d": [20,40,60,40], "sS": sS, 
                    "replications": 1000}
        print(Policies.simulate_sS(**instance))

if __name__ == '__main__':
    Policies.simulate_sample_policy()