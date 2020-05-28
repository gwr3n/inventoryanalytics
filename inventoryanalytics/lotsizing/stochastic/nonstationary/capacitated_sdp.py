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

class Demand:
    def pmf(self) -> List[float]:
        return self.pmf

class PoissonDemand(Demand):
    def __init__(self, d: List[float], q: float):
        """Creates an instance of PoissonDemand

        Arguments:
            d {List[float]} -- the demand rates
            q {float} -- quantile truncation for the demand
        """
        self.d = d
        self.max_demand = lambda d: sp.poisson(d).ppf(q).astype(int)                           # max demand in the support
        self.p = lambda d, k : sp.poisson(d).pmf(k)/q                                          # poisson pmf
        self.pmf = [[[k, self.p(d, k)] for k in range(0, self.max_demand(d))] for d in self.d] # pmf

class NormalDemand(Demand):
    def __init__(self, d: List[float], cv: float, q: float):
        """[summary]

        Arguments:
            d {List[float]} -- the demand mean
            cv {float} -- the coefficient of variation
            q {float} -- quantile truncation for the demand
        """
        self.d, self.cv  = d, cv
        self.max_demand = lambda d: sp.norm(d, cv*d).ppf(q).astype(int)                        # max demand in the support
        self.p = lambda d, k : (sp.norm(d, cv*d).cdf(k+0.5) - \
                                sp.norm(d, cv*d).cdf(k-0.5)) / \
                               (q+0.5-sp.norm(d, cv*d).cdf(-0.5))                              # normal pmf
        self.pmf = [[[k, self.p(d, k)] for k in range(0, self.max_demand(d))] for d in self.d] # pmf

class PmfDemand(Demand):
    def __init__(self, pmf: List[float]):
        """Creates an instance of PmfDemand from a pmf

        Arguments:
            pmf {List[float]} -- the pmf expressed as [[0,0.5],[5,0.5]] where in [5,0.5] 
                                 5 is the demand and 0.5 the associated probabilty mass
        """
        self.pmf = pmf

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

    def __init__(self, K: float, B: float, v: float, h: float, p: float, w: float, d: Demand, 
                 max_inv: float, initial_order: bool):
        """
        Create an instance of StochasticLotSizing.
        
        Arguments:
            K {float} -- the fixed ordering cost
            B {float} -- the fixed order capacity
            v {float} -- the proportional unit ordering cost
            h {float} -- the proportional unit inventory holding cost
            p {float} -- the proportional unit inventory penalty cost
            w {float} -- the discount factor
            d {Demand} -- the demand
            max_inv {float} -- the maximum inventory level
            initial_order {bool} -- allow order in the first period
        """

        # initialize instance variables
        self.T, self.K, self.B, self.v, self.h, self.p, self.w, self.max_inv = len(d.pmf), K, B, v, h, p, w, max_inv
        self.pmf = d.pmf

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
                       (self.w*self._f(self.st(s, a, p[0])) if s.t < self.T-1 else 0))   # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions

        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+                       # optimal action
                                    (self.w*self._f(self.st(s, a, p[0])) if s.t < self.T-1 else 0)) 
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
        for t in range(0, len(self.pmf)):
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
    
    # Shiaoxiang 1996
    # instance = {"K": 22, "B": 9, "v": 0, "h": 1, "p": 10, "w": 0.9, 
    #             "d": PmfDemand([[[6, 0.95], [7, 0.05]] for k in range(20)]), 
    #             "max_inv": 300, "initial_order": False}

    # Gallego 2000
    # instance = {"K": 10, "B": 10, "v": 0, "h": 1, "p": 9, "w": 0.9, 
    #             "d": PmfDemand([[[1, 0.15], [6, 0.70], [7, 0.15]] for k in range(52)]), 
    #             "max_inv": 300, "initial_order": False}
    
    # Rossi et al. 2020
    instance = {"K": 100, "B": 65, "v": 0, "h": 1, "p": 10, "w": 1, "d": PoissonDemand([20,40,60,40],0.9999), "max_inv": 300, "initial_order": False}
    # instance = {"K": 100, "B": 65, "v": 0, "h": 1, "p": 10, "w": 1, "d": NormalDemand([20,40,60,40], 0.25 ,0.9999), "max_inv": 300, "initial_order": False}
    
    # This cycle also builds the internal map
    lot_sizing_no_order = StochasticLotSizing(**instance)
    t = 0                       # initial period
    for i in range(*domain):    # range over inventory level domain
        print("Optimal policy cost: "    + str(lot_sizing_no_order.f(i)))
    
    # Plot G(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_no_order.f(k) for k in range(*domain)], label='G(y)')
    # Print G
    # print(list(zip([k for k in range(*domain)], [lot_sizing_no_order.f(k) for k in range(*domain)])))

    # Build action map
    instance["initial_order"]=True
    lot_sizing_order = StochasticLotSizing(**instance)
    for i in range(*domain):    # range over inventory level domain
        print("Optimal policy cost: "    + str(lot_sizing_order.f(i)))
        print("Optimal order quantity("+str(i)+"): " + str(lot_sizing_order.q(t, i)))

    # Plot C(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_order.f(k) for k in range(*domain)], label='C(y)')
    # Print C
    # print(list(zip([k for k in range(*domain)], [lot_sizing_order.f(k) for k in range(*domain)])))
    
    # Plot C(y)-G(y)
    plt.plot([k for k in range(*domain)], [lot_sizing_order.f(k) - lot_sizing_no_order.f(k) for k in range(*domain)], 
             label='C(y)-G(y)')
    # print(list(zip([k for k in range(*domain)], [lot_sizing_order.f(k) - lot_sizing_no_order.f(k) for k in range(*domain)])))

    #Plot Q
    plt.scatter([k for k in range(*domain)], [lot_sizing_order.q(0,k) for k in range(*domain)], s=2, label='Q')
    plt.ylabel('Optimal policy cost')

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

    plt.legend()
    plt.show()