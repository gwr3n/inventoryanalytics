from typing import List
from inventoryanalytics.utils import memoize as mem
import scipy.stats as sp

class State:
    """
    The state of the inventory system.
    
    Returns:
        [type] -- state of the inventory system
    """

    def __init__(self, t: int, I: float):
        self.t, self.I = t, I

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.t) + " " + str(self.I)

    def __hash__(self):
        return hash(str(self))

class StochasticLotSizing:
    """
    The nonstationary stochastic lot sizing problem.

    Herbert E. Scarf. Optimality of (s,S) policies in the 
    dynamic inventory problem. In K. J. Arrow, S. Karlin, 
    and P. Suppes, editors, Mathematical Methods in the 
    Social Sciences, pages 196–202. Stanford University 
    Press, Stanford, CA, 1960.

    Returns:
        [type] -- A problem instance
    """


    M = 200         #max inventory
    qt = 0.9999     #quantile_truncation

    def __init__(self, K: float, v: float, h: float, p: float, C: float, d: List[float]):
        """
        Create an instance of StochasticLotSizing.
        
        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the proportional unit ordering cost
            h {float} -- the proportional unit inventory holding cost
            p {float} -- the proportional unit inventory penalty cost
            C {float} -- the ordering capacity
            d {List[float]} -- the demand probability mass function 
              taking the form [[d_1,p_1],...,[d_N,p_N]], where d_k is 
              the k-th value in the demand support and p_k is its 
              probability.
        """
        #placeholders
        max_inv, q = StochasticLotSizing.M, StochasticLotSizing.qt      # max inventory level
        max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)         # max demand in the support
        
        #initialize instance variables
        self.T, self.K, self.v, self.h, self.p, self.C, self.d = len(d)-1, K, v, h, p, C, d
        
        pmf = lambda d, k : sp.poisson(d).pmf(k)/q                      # poisson pmf
        self.pmf = [[[k, pmf(d, k)] for k in range(0, max_demand(d))] for d in self.d]
        self.ag = lambda s: [x for x in range(0, max_inv-s.I)]          # action generator
        self.st = lambda s, a, d: State(s.t+1, s.I+a-d)                 # state transition
        
        L = lambda i,a,d : self.h*max(i+a-d, 0) + self.p*max(d-i-a, 0)  # immediate holding/penalty cost
        self.iv = lambda s, a, d: (self.K if a > 0 else 0) + L(s.I, a, d) # immediate value function

        self.cache_actions = {}                                         # cache with optimal state/action pairs

    def f(self, level: float) -> float:
        """
        Recursively solve the nonstationary stochastic lot sizing problem
        for an initial inventory level.
        
        Arguments:
            level {float} -- the initial inventory level
        
        Returns:
            float -- the cost of an optimal policy 
        """

        s = State(0,level)
        return self.__f(s)
    
    def q(self, level:float) -> float:
        """
        Retrieves the optimal order quantity for a given initial inventory level.
        Function :func:`f` must have been called before using this method.

        Arguments:
            level {float} -- the initial inventory level
        
        Returns:
            float -- the optimal order quantity 
        """

        s = State(0,level)
        return self.cache_actions[str(s)]

    @mem.memoize
    def __f(self, s: State) -> float:
        """
        Dynamic programming forward recursion.
        
        Arguments:
            s {State} -- the initial state
        
        Returns:
            float -- the cost of an optimal policy 
        """
        #Forward recursion
        v = min(
            [sum([p[1]*(self.iv(s, a, p[0])+                                    # immediate cost
                       (self.__f(self.st(s, a, p[0])) if s.t < self.T else 0))  # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions
        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+
                                    (self.__f(self.st(s, a, p[0])) if s.t < self.T else 0)) 
                               for p in self.pmf[s.t]]) == v          
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost

if __name__ == '__main__':
    instance = {"K": 100, "v": 0, "h": 1, "p": 10, "C": None, "d": [20,40,60,40]}
    lot_sizing = StochasticLotSizing(**instance)
    initial_inventory_level = 0
    print("Optimal policy cost: "    + str(lot_sizing.f(initial_inventory_level)))
    print("Optimal order quantity: " + str(lot_sizing.q(initial_inventory_level)))