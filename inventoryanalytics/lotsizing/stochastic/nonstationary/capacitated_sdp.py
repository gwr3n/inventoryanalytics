'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
import scipy.stats as sp
import json
import matplotlib.pyplot as plt
import numpy.random as rnd

import functools 

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

class State:
    """
    The state of the inventory system.
    
    Returns:
        [type] -- state of the inventory system
    """

    def __init__(self, t: int, I: float):
        """[summary]
        
        Arguments:
            t {int} -- the time period
            I {float} -- the initial inventory
        """

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

    def __init__(self, K: float, B: float, v: float, h: float, p: float, d: List[float], 
                 max_inv: float, q: float, initial_order: bool):
        """
        Create an instance of StochasticLotSizing.
        
        Arguments:
            K {float} -- the fixed ordering cost
            B {float} -- the fixed order capacity
            v {float} -- the proportional unit ordering cost
            h {float} -- the proportional unit inventory holding cost
            p {float} -- the proportional unit inventory penalty cost
            d {List[float]} -- the mean demand (assume Poisson distribution).
            max_inv {float} -- the maximum inventory level
            q {float} -- quantile truncation for the demand
            initial_order {bool} -- allow order in the first period
        """

        # placeholders
        max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)         # max demand in the support
        #cv = 0.15                                                        # coefficient of variation
        #max_demand = lambda d: sp.norm(d, cv*d).ppf(q).astype(int)       # max demand in the support
        
        # initialize instance variables
        self.T, self.K, self.B, self.v, self.h, self.p, self.d, self.max_inv = len(d)-1, K, B, v, h, p, d, max_inv
        pmf = lambda d, k : sp.poisson(d).pmf(k)/q                      # poisson pmf
        #pmf = lambda d, k : (sp.norm(d, cv*d).cdf(k+0.5) - \
        #                     sp.norm(d, cv*d).cdf(k-0.5)) / \
        #                    (q-sp.norm(d, cv*d).cdf(-0.5)) # normal pmf
        self.pmf = [[[k, pmf(d, k)] for k in range(0, max_demand(d))] for d in self.d]

        # lambdas
        if initial_order:                                               # action generator
            self.ag = lambda s: [x for x in range(0, min(max_inv-s.I, self.B+1))]      
        else: 
            self.ag = lambda s: [x for x in range(0, min(max_inv-s.I, self.B+1))] if s.t > 0 else [0] 
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

    @memoize
    def _f(self, s: State) -> float:
        """
        Dynamic programming forward recursion.
        
        Arguments:
            s {State} -- the initial state
        
        Returns:
            float -- the cost of an optimal policy 
        """
        #Forward recursion
        v = min(                                                                # optimal cost
            [sum([p[1]*(self.iv(s, a, p[0])+                                    # immediate cost
                       (self._f(self.st(s, a, p[0])) if s.t < self.T else 0))   # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions

        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+                       # optimal action
                                    (self._f(self.st(s, a, p[0])) if s.t < self.T else 0)) 
                               for p in self.pmf[s.t]]) == v          
                               
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost

    def extract_sS_policy(self, min_inv) -> List[float]:
        """
        Extract optimal (sk,Sk) policy parameters
        
        Returns:
            List[float] -- the optimal sk,Sk policy parameters [...,[s_k,S_k],...]
        """

        for i in range(min_inv, self.max_inv):
            self.f(i)
        policy_parameters = []
        for t in range(0, len(self.d)):
            policy_parameters.append([])
            level, min_level = self.max_inv - 2, min_inv
            s, nextState = State(t, level), State(t, level+1)
            while level > min_level:
                if (self.cache_actions.get(str(s), 0) > 0 and self.cache_actions.get(str(nextState), 0) == 0) or \
                   (self.cache_actions.get(str(nextState), 0) > self.cache_actions.get(str(s), 0)):
                    policy_parameters[t].append([level, level+self.cache_actions.get(str(s), 0)])
                level, s, nextState = level - 1, State(t, level - 1), State(t, level)
        return policy_parameters

    def testKBConvexity(self, min_inv) -> bool:
        step, k = 1, 0
        while k < 1000:
            x = rnd.randint(min_inv, self.max_inv-1)
            a = rnd.randint(0,min(self.max_inv-1-x,self.B))      
            gx, gxa, gxd = self.f(x), self.f(x+a), self.f(x+step) - self.f(x)
            if self.K + gxa - gx - a*gxd < 0:
                print("K:" + str(self.K))
                print("x:" + str(x))
                print("a:" + str(a))
                print("gx:" + str(gx))
                print("gxa:" + str(gxa))
                print("gxd:" + str(gxd))
                print("Discrepancy: "+str(self.K + gxa - gx - a*gxd))
                print("If discrepancy is very low, this may be due to rounding errors.")
                return False
            else:
                k += 1
        return True
    
    def testAlwaysOrder(self, domain) -> bool:
        t = 0
        orders = [lot_sizing_order.q(t, i) for i in range(*domain)]
        while(orders[-1]==0):
            orders.pop(-1)
        return orders.count(0) == 0 

if __name__ == '__main__':
    domain = (-20,200)          # inventory level domain for plotting
    instance = {"K": 100, "B": 65, "v": 0, "h": 1, "p": 10, "d": [20,40,60,40],
                "max_inv": 300, "q": 0.999, "initial_order": False}
    
    # This cycle also builds the internal map
    lot_sizing_no_order = StochasticLotSizing(**instance)
    t = 0                       # initial period
    for i in range(*domain):    # range over inventory level domain
        print("Optimal policy cost: "    + str(lot_sizing_no_order.f(i)))
    
    # Plot G(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_no_order.f(k) for k in range(*domain)])

    # Build action map
    instance["initial_order"]=True
    lot_sizing_order = StochasticLotSizing(**instance)
    for i in range(*domain):    # range over inventory level domain
        print("Optimal policy cost: "    + str(lot_sizing_order.f(i)))
        print("Optimal order quantity("+str(i)+"): " + str(lot_sizing_order.q(t, i)))
    #Plot Q
    plt.plot([k for k in range(*domain)], [lot_sizing_order.q(0,k) for k in range(*domain)])
    plt.ylabel('Optimal policy cost')

    # Plot C(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_order.f(k) for k in range(*domain)])
    
    # Plot C(y)-G(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_order.f(k) - lot_sizing_no_order.f(k) for k in range(*domain)])

    # Print G
    #print(list(zip([k for k in range(*domain)], [lot_sizing_no_order.f(k) for k in range(*domain)])))

    # Extract [sk,Sk] Policy 
    print()
    print("****** [sk,Sk] Policy *****")
    min_inv = -50               # min inventory level
    skSk_policy = lot_sizing_order.extract_sS_policy(min_inv)   
    period = 1
    for i in skSk_policy:
        print("Period: "+str(period)+"[sk,Sk]")
        for j in i:
            print(j)
        period += 1
        print()
    print("***************************")

    print()
    print("***************************")
    try:
        print("The function is (K,B)-convex") if lot_sizing_no_order.testKBConvexity(min_inv) else print("The function is not (K,B)-convex")
    except Exception as e:
        print("(K,B)-convexity test failed")
        print(str(e))
    print("***************************")

    print()
    print("***************************")
    try:
        print("The function satisfies the AlwaysOrder property") if lot_sizing_no_order.testAlwaysOrder(domain) else print("The function does not satisfy the AlwaysOrder property")
    except Exception as e:
        print("AlwaysOrder test failed")
        print(str(e))
    print("***************************")

    plt.show()