from typing import List
from inventoryanalytics.utils import memoize as mem
import scipy.stats as sp
import json

class State:

    def __init__(self, t: int, I: List[float]):
        self.t, self.I = t, I

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.t) + " " + str(self.I)

    def __hash__(self):
        return hash(str(self))

class MultiItemStochasticLotSizing:

    def __init__(self, K: float, v: float, h: float, p: float, d: List[float], 
                 max_inv: float, q: float, initial_order: bool):
        # placeholders
        max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)         # max demand in the support
        
        # initialize instance variables
        self.T, self.K, self.v, self.h, self.p, self.d, self.max_inv = len(d)-1, K, v, h, p, d, max_inv
        pmf = lambda d, k : sp.poisson(d).pmf(k)/q                      # poisson pmf
        self.pmf = [[[(i,j), pmf(d, i)*pmf(d, j)] for i in range(0, max_demand(d)) for j in range(0, max_demand(d))] for d in self.d]

        # lambdas
        if initial_order:                                               # action generator
            self.ag = lambda s: [(i,j) for i in range(0, max_inv-s.I[0]) for j in range(0, max_inv-s.I[1])]      
        else: 
            self.ag = lambda s: [(i,j) for i in range(0, max_inv-s.I[0]) for j in range(0, max_inv-s.I[1])]  if s.t > 0 else [(0,0)] 
        self.st = lambda s, a, d: State(s.t+1, (s.I[0]+a[0]-d[0],s.I[1]+a[1]-d[1]))                 # state transition
        L = lambda i,a,d : (self.h*max(i[0]+a[0]-d[0], 0) + self.p*max(d[0]-i[0]-a[0], 0)) + (self.h*max(i[1]+a[1]-d[1], 0) + self.p*max(d[1]-i[1]-a[1], 0))  # immediate holding/penalty cost
        self.iv = lambda s, a, d: (self.K if sum(a) > 0 else 0) + L(s.I, a, d) # immediate value function

        self.cache_actions = {}                                         # cache with optimal state/action pairs

    def f(self, level: List[float]) -> float:
        s = State(0,level)
        return self._f(s)

    def q(self, period: int, level: List[float]) -> float:
        s = State(period,level)
        return self.cache_actions[str(s)]

    @mem.memoize
    def _f(self, s: State) -> float:
        #Forward recursion
        v = min(
            [sum([p[1]*(self.iv(s, a, p[0])+                                    # immediate cost
                       (self._f(self.st(s, a, p[0])) if s.t < self.T else 0))   # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions
        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+
                                    (self._f(self.st(s, a, p[0])) if s.t < self.T else 0)) 
                               for p in self.pmf[s.t]]) == v          
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost