from typing import List
from inventoryanalytics.utils import memoize as mem
import scipy.stats as sp

'''
The nonstationary stochastic lot sizing problem.

Herbert E. Scarf. Optimality of (s,S) policies in the 
dynamic inventory problem. In K. J. Arrow, S. Karlin, 
and P. Suppes, editors, Mathematical Methods in the 
Social Sciences, pages 196–202. Stanford University 
Press, Stanford, CA, 1960.
'''

class StochasticLotSizing:

    maxInv = 200     #max inventory
    qt = 0.9999      #quantile_truncation
    
    def __init__(self, K: float, v: float, h: float, p: float, C: float, d: List[float]):
        self.T, self.K, self.v, self.h, self.p, self.C, self.d, q = len(d)-1, K, v, h, p, C, d, StochasticLotSizing.qt
        self.pmf = [[[k, sp.poisson(d).pmf(k)/q] for k in range(0,sp.poisson(d).ppf(q).astype(int))] for d in self.d]
        self.ag = lambda s: [x for x in range(0,StochasticLotSizing.maxInv-s.I)] # action generator
        self.st = lambda s, a, d: self.State(s.t+1,s.I+a-d) # state transition
        self.iv = lambda s, a, d: (self.K if a > 0 else 0) + self.h*max(s.I+a-d,0) + self.p*max(d-s.I-a,0) # immediate value
        self.cache_actions = {}

    def f(self, level: float) -> float:
        '''
        Recursively solve the nonstationary stochastic lot sizing problem
        for an initial inventory level.
        '''
        s = self.State(0,level)
        return self.__f(s)
    
    def q(self, level:float) -> float:
        '''
        Retrieves the optimal order quantity for a given initial inventory level.
        Function :func:`f` must have been called before using this method.
        '''
        s = self.State(0,level)
        for k in self.cache_actions[str(s)]: 
            return k
        return None

    @mem.memoize
    def __f(self, s) -> float:
        v = min([sum([p[1]*(self.iv(s,a,p[0])+(self.__f(self.st(s,a,p[0])) if s.t < self.T else 0)) for p in self.pmf[s.t]]) for a in self.ag(s)])
        q = filter(lambda a: sum([p[1]*(self.iv(s,a,p[0])+(self.__f(self.st(s,a,p[0])) if s.t < self.T else 0)) for p in self.pmf[s.t]]) == v, self.ag(s))
        self.cache_actions[str(s)]=q
        return v 

    class State:

        def __init__(self, t: int, I: float):
            self.t, self.I = t, I

        def __eq__(self, other): 
            return self.__dict__ == other.__dict__

        def __str__(self):
            return str(self.t) + " " + str(self.I)

        def __hash__(self):
            return hash(str(self))    

if __name__ == '__main__':
    instance = {"K": 100, "v": 0, "h":1, "p":10, "C":None, "d":[20,40,60,40]}
    lot_sizing = StochasticLotSizing(**instance)
    level = 0
    print("Optimal policy cost: " + str(lot_sizing.f(level)))
    print("Optimal order quantity: " + str(lot_sizing.q(level)))