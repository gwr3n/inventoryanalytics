'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
from inventoryanalytics.utils import memoize as mem
import scipy.stats as sp
import json

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

    Returns:
        [type] -- A problem instance
    """

    def __init__(self, K: float, v: float, h: float, p: float, d: List[float], 
                 max_inv: float, q: float, initial_order: bool):
        """
        Create an instance of StochasticLotSizing.
        
        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the proportional unit ordering cost
            h {float} -- the proportional unit inventory holding cost
            p {float} -- the proportional unit inventory penalty cost
            d {List[float]} -- the demand probability mass function 
              taking the form [[d_1,p_1],...,[d_N,p_N]], where d_k is 
              the k-th value in the demand support and p_k is its 
              probability.
            max_inv {float} -- the maximum inventory level
            q {float} -- quantile truncation for the demand
            initial_order {bool} -- allow order in the first period
        """
        # placeholders
        max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)         # max demand in the support
        
        # initialize instance variables
        self.T, self.K, self.v, self.h, self.p, self.d, self.max_inv = len(d)-1, K, v, h, p, d, max_inv
        pmf = lambda d, k : sp.poisson(d).pmf(k)/q                      # poisson pmf
        self.pmf = [[[k, pmf(d, k)] for k in range(0, max_demand(d))] for d in self.d]

        # lambdas
        if initial_order:                                               # action generator
            self.ag = lambda s: [x for x in range(0, max_inv-s.I)]      
        else: 
            self.ag = lambda s: [x for x in range(0, max_inv-s.I)] if s.t > 0 else [0] 
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
        return self._f(s)
    
    def q(self, period: int, level:float) -> float:
        """
        Retrieves the optimal order quantity for a given initial inventory level.
        Function :func:`f` must have been called before using this method.

        Arguments:
            period {int} -- the initial period
            level {float} -- the initial inventory level
        
        Returns:
            float -- the optimal order quantity 
        """

        s = State(period,level)
        return self.cache_actions[str(s)]

    def extract_sS_policy(self) -> List[float]:
        """
        Extract optimal (s,S) policy parameters

        Herbert E. Scarf. Optimality of (s,S) policies in the 
        dynamic inventory problem. In K. J. Arrow, S. Karlin, 
        and P. Suppes, editors, Mathematical Methods in the 
        Social Sciences, pages 196–202. Stanford University 
        Press, Stanford, CA, 1960.
        
        Returns:
            List[float] -- the optimal s,S policy parameters [...,[s_k,S_k],...]
        """

        for i in range(-self.max_inv, self.max_inv):
            self.f(i)
        policy_parameters = []
        for t in range(0, len(self.d)):
            level = self.max_inv - 1
            min_level = -self.max_inv
            s = State(t, level)
            while self.cache_actions.get(str(s), 0) == 0 and level > min_level:
                level, s = level - 1, State(t, level - 1)
            policy_parameters.append(
                [level, level+self.cache_actions.get(str(s), 0)])
        return policy_parameters

    @mem.memoize
    def _f(self, s: State) -> float:
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
                       (self._f(self.st(s, a, p[0])) if s.t < self.T else 0))   # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions
        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+
                                    (self._f(self.st(s, a, p[0])) if s.t < self.T else 0)) 
                               for p in self.pmf[s.t]]) == v          
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost

    @staticmethod
    def run_instance(file_name: str = None):
        instance = {"K": 100, "v": 0, "h": 1, "p": 10, "d": [20,40,60,40], 
                    "max_inv": 200, "q": 0.9999, "initial_order": True}
        lot_sizing = StochasticLotSizing(**instance)
        t = 0   # initial period
        i = 0   #initial inventory level
        print("Optimal policy cost: "    + str(lot_sizing.f(i)))
        print("Optimal order quantity: " + str(lot_sizing.q(t, i)))
        print(lot_sizing.extract_sS_policy())

        try:
            with open(file_name, 'w') as f:
                json.dump(lot_sizing.cache_actions, f)
                f.close()
                print("Policy saved to "+file_name) 
        except:
            print("Provide a file name to save the policy to disk.") 

    @staticmethod
    def run_instance_stationary():
        instance = {"K": 64, "v": 0, "h": 1, "p": 9, "d": [10,10,10,10,10,10,10], 
                    "max_inv": 200, "q": 0.9999, "initial_order": True}
        lot_sizing = StochasticLotSizing(**instance)
        print(lot_sizing.extract_sS_policy())

if __name__ == '__main__':
    StochasticLotSizing.run_instance()