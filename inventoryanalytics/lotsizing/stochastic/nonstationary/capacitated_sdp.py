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
import signal

import functools 
import time

class TimeoutException(Exception):
    pass

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def deadline(seconds, *args):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutException("Timed out!")

        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return f(*args)
            finally:
                signal.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate

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
                 min_inv: float, max_inv: float, initial_order: bool):
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
            min_inv {float} -- the minimum inventory level
            max_inv {float} -- the maximum inventory level
            initial_order {bool} -- allow order in the first period
        """

        # initialize instance variables
        self.T, self.K, self.B, self.v, self.h = len(d.pmf), K, B, v, h
        self.p, self.w, self.pmf, self.min_inv, self.max_inv = p, w, d.pmf, min_inv, max_inv

        # lambdas
        if initial_order:                                               # action generator
            self.ag = lambda s: [x for x in range(0, min(max_inv-s.I, self.B+1))]      
        else: 
            self.ag = lambda s: [x for x in range(0, min(max_inv-s.I, self.B+1))] if s.t > 0 else [0] 
        self.st = lambda s, a, d: State(s.t+1, s.I+a-d)                 # state transition
        L = lambda i,a,d : self.h*max(i+a-d, 0) + self.p*max(d-i-a, 0)  # immediate holding/penalty cost
        self.iv = lambda s, a, d: (self.K+v*a if a > 0 else 0) + L(s.I, a, d) # immediate value function

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
                       (self.w*self._f(self.st(s, a, p[0])) if s.t < self.T-1 else 0)) # future cost
                  for p in self.pmf[s.t]])                                      # demand realisations
             for a in self.ag(s)])                                              # actions

        opt_a = lambda a: sum([p[1]*(self.iv(s, a, p[0])+                       # optimal action
                                    (self.w*self._f(self.st(s, a, p[0])) if s.t < self.T-1 else 0)) 
                               for p in self.pmf[s.t]]) == v          
                               
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost

    def extract_skSk_policy(self) -> List[float]:
        """
        Extract optimal (sk,Sk) policy parameters
        
        Returns:
            List[float] -- the optimal sk,Sk policy parameters [...,[s_k,S_k],...]
        """

        for i in range(self.min_inv, self.max_inv):
            self.f(i)
        policy_parameters = []
        for t in range(0, len(self.pmf)):
            policy_parameters.append([])
            level, min_level = self.max_inv - 2, self.min_inv
            s, nextState = State(t, level), State(t, level+1)
            while level > min_level:
                if (self.cache_actions.get(str(s), 0) > 0 and self.cache_actions.get(str(nextState), 0) == 0) or \
                   (self.cache_actions.get(str(nextState), 0) > self.cache_actions.get(str(s), 0)):
                    policy_parameters[t].append([level, level+self.cache_actions.get(str(s), 0)])
                level, s, nextState = level - 1, State(t, level - 1), State(t, level)
        return policy_parameters

    @timer
    def testKBConvexity(self) -> bool:
        try:
            return self.testKBConvexity_full()
        except TimeoutException:
            print("testKBConvexity_full() has timed out, switching to random checks...")
            return self.testKBConvexity_random()

    @deadline(60)                                         #KBConvexity check timeout in seconds
    def testKBConvexity_full(self) -> bool:
        """
        Tests CK convexity as originally introduced in Gallego G, Scheller-Wolf A (2000) 
        Capacitated inventory problems with xed order costs: Some optimal policy structure. 
        European Journal of Operational Research 126(3):603-613.

        Returns:
            bool -- true if the function is CK convex
        """
        return all([self.K + self.f(x+a) - self.f(x) - a*(self.f(y) - self.f(y-b))/b >= 0
                    for x in range(self.min_inv+1, self.max_inv-1) 
                    for y in range(self.min_inv+1, x-1)
                    for a in range(1, min(self.B+1,self.max_inv-x))
                    for b in range(1, min(self.B+1,y-self.min_inv))])

    def testKBConvexity_random(self) -> bool:
        """
        Tests CK convexity as originally introduced in Gallego G, Scheller-Wolf A (2000) 
        Capacitated inventory problems with xed order costs: Some optimal policy structure. 
        European Journal of Operational Research 126(3):603-613.

        Returns:
            bool -- true if the function is CK convex
        """
        N = 1000
        print("Running "+str(N)+" random checks...")
        step, k = 1, 0
        while k < N:
            x = rnd.randint(self.min_inv, self.max_inv-1)
            a = rnd.randint(0,min(self.max_inv-1-x,self.B))      
            gx, gxa, gxd = self.f(x), self.f(x+a), self.f(x) - self.f(x-step)
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
    
    @timer
    def testAlwaysOrder(self) -> bool:
        """Tests if there exists a threshold s such that no order is placed for inventory greater than s
        and an order is placed for inventory less or equal to s

        Returns:
            bool -- true if the function satisfies the AlwaysOrder property
        """

        t = 0
        orders = [lot_sizing_order.q(t, i) for i in range(self.min_inv,self.max_inv)]
        while(orders[-1]==0):
            orders.pop(-1)
        return orders.count(0) == 0 

if __name__ == '__main__':
    plot_domain = (-20,200)          # inventory level domain for plotting and printing
    plotFunctions = True
    printFunctions = True
    
    # Shiaoxiang 1996
    # instance = {"K": 22, "B": 9, "v": 0, "h": 1, "p": 10, "w": 0.9, 
    #             "d": PmfDemand([[[6, 0.95], [7, 0.05]] for k in range(20)]), 
    #             "min_inv": -10, "max_inv": 300, "initial_order": False}

    # Gallego 2000
    # instance = {"K": 10, "B": 10, "v": 0, "h": 1, "p": 9, "w": 0.9, 
    #             "d": PmfDemand([[[1, 0.15], [6, 0.70], [7, 0.15]] for k in range(52)]), 
    #             "min_inv": -50, "max_inv": 300, "initial_order": False}
    
    # Rossi et al. 2020
    instance = {"K": 100, "B": 65, "v": 0, "h": 1, "p": 10, "w": 1, "d": PoissonDemand([20,40,60,40],0.9999), "min_inv": -50, "max_inv": 300, "initial_order": False}
    # instance = {"K": 100, "B": 65, "v": 0, "h": 1, "p": 10, "w": 1, "d": NormalDemand([20,40,60,40], 0.25 ,0.9999), "min_inv": -50, "max_inv": 300, "initial_order": False}

    # Define G(y) and C(y)
    lot_sizing_no_order = StochasticLotSizing(**instance)   # G(y)
    instance["initial_order"]=True
    lot_sizing_order = StochasticLotSizing(**instance)      # C(y)

    
    if plotFunctions: 
        plt.xlabel("$x$")
        plt.plot([k for k in range(*plot_domain)],              # Plot G(y)
                 [lot_sizing_no_order.f(k) 
                 for k in range(*plot_domain)], label='$G(x)$')
    
        plt.plot([k for k in range(*plot_domain)],              # Plot C(x)
                 [lot_sizing_order.f(k) 
                 for k in range(*plot_domain)], label='$C(x)$')
    
        plt.plot([k for k in range(*plot_domain)],              # Plot C(y)-G(y)
                 [lot_sizing_order.f(k) - lot_sizing_no_order.f(k) 
                 for k in range(*plot_domain)], label='C(y)-G(y)')

        plt.scatter([k for k in range(*plot_domain)],           #Plot Q
                    [lot_sizing_order.q(0,k) for k in range(*plot_domain)], s=2, label='$Q(x)$', color='red')

        plt.legend()
        #plt.savefig('/Users/gwren/Downloads/12_scarf_G.svg', format='svg')
    
    if printFunctions: 
        print("G(y) = "+str(list(zip([k for k in range(*plot_domain)],        # Print G(y)
                                     [lot_sizing_no_order.f(k) for k in range(*plot_domain)]))))
        print()
        print("C(y) = "+str(list(zip([k for k in range(*plot_domain)],        # Print C(y)
                                     [lot_sizing_order.f(k) for k in range(*plot_domain)]))))
        print()
        print("C(y)-G(y) = "+str(list(zip([k for k in range(*plot_domain)],   # Print C(y)-G(y)
                                          [lot_sizing_order.f(k) - lot_sizing_no_order.f(k) 
                                           for k in range(*plot_domain)]))))
        print()
        print("Q = "+str(list(zip([k for k in range(*plot_domain)],           # Print Q
                                  [lot_sizing_order.q(0,k) for k in range(*plot_domain)]))))

    # Extract [sk,Sk] Policy 
    print()
    print("****** [sk,Sk] Policy *****")
    skSk_policy = lot_sizing_order.extract_skSk_policy()   
    period = 1
    for i in range(len(skSk_policy)):
        print("Period: "+str(i)+"\n[sk,Sk]")
        for j in skSk_policy[i]:
            print(j)
        period += 1
        print()
    print("***************************")

    print()
    print("********** Checks *********")

    try:
        if lot_sizing_no_order.testKBConvexity():           # Decorate this function to set a suitable timeout in seconds
            print("The function G(y) is (K,B)-convex")  
        else: 
            print("The function G(y) is not (K,B)-convex")
    except Exception as e:
        print("(K,B)-convexity test for G(y) failed")
        print(str(e))

    print()
    try:
        if lot_sizing_no_order.testAlwaysOrder():
            print("The function satisfies the AlwaysOrder property")  
        else: 
            print("The function does not satisfy the AlwaysOrder property")
    except Exception as e:
        print("AlwaysOrder test failed")
        print(str(e))
    print("***************************")

    plt.show()