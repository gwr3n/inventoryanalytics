'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from docplex.mp.model import Model
import sys

# sys.path.insert(0,'/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx')
sys.path.insert(0,'/Applications/CPLEX_Studio1210/cplex/python/3.7/x86-64_osx')

# http://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
# http://www-01.ibm.com/support/docview.wss?uid=swg27042869&aid=1

from typing import List
import functools

class CapacitatedLotSizing:
    """
    A capacitated lot sizing problem under capacity constraints.

    M. Florian, J. K. Lenstra, and A. H. G. Rinnooy Kan. 
    Deterministic production planning: Algorithms and complexity. 
    Management Science, 26(7): 669-679, July 1980
    """
    def __init__(self, K: float, v: float, h: float, d: List[float], I0: float, C: float):
        """
        Create an instance of the capacitated lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the per unit ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
            C {float} -- the order capacity
        """

        self.K, self.v, self.h, self.d, self.I0, self.C = K, v, h, d, I0, C

class CapacitatedLotSizingCPLEX(CapacitatedLotSizing):
    """
    Solves the capacitated lot sizing problem as an MILP.
    """

    def __init__(self, K: float, v: float, h: float, d: List[float], I0, C: float):
        """
        Create an instance of the capacitated lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the per unit ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """
        super().__init__(K, v, h, d, I0, C)
        self.model()

    def model(self):
        """
        Model and solve the capacitated lot sizing problem via CPLEX
        """

        model = Model("Capacitated lot sizing")
        T = len(self.d)
        idx = [t for t in range(0,T)]
        self.Q = model.continuous_var_dict(idx, name="Q")
        I = model.continuous_var_dict(idx, lb=0, name="I")
        delta = model.binary_var_dict(idx, name="delta")
        

        for t in range(0,T):
            model.add_constraint(self.Q[t] <= delta[t]*self.C)
            model.add_constraint(self.I0 + model.sum(self.Q[k] - self.d[k] for k in range(0,t+1)) == I[t])
            model.add_constraint(self.Q[t] >= 0)
            model.add_constraint(I[t] >= 0)

        model.minimize(model.sum(delta[t] * self.K + self.Q[t] * self.v + self.h * I[t] for t in range(0,T)))
        model.print_information()
        self.msol = model.solve()
        if self.msol:
            model.print_solution()
        else:
            print("Solve status: " + self.msol.get_solve_status() + "\n")
        
    def order_quantities(self) -> List[float]:
        '''
        Compute optimal capacitated lot sizing order quantities
        '''
        return [self.msol.get_var_value(self.Q[t]) for t in range(0,len(self.d))]
    
    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the capacitated lot sizing problem
        '''
        return self.msol.get_objective_value()    

    @staticmethod
    def _test():
        print("********** CapacitatedLotSizingCPLEX **********")
        instance = {"K": 40, "v": 1, "h": 1, "d":[10,20,30,40], "I0": 0, "C": 30}
        CapacitatedLotSizingCPLEX(**instance)

class memoize(object): 
    """
    Memoization utility
    """

    def __init__(self, func): 
        self.func, self.memoized, self.method_cache = func, {}, {}

    def __call__(self, *args): 
        return self.cache_get(self.memoized, args, lambda: self.func(*args)) 

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
        self.memoized, self.method_cache = {}, {}

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

class CapacitatedLotSizingSDP(CapacitatedLotSizing):
    """
    Solves the capacitated lot sizing problem as an SDP.
    """

    def __init__(self, K: float, v: float, h: float, d: List[float], I0, C: float):
        """
        Create an instance of the capacitated lot sizing problem.

        Arguments:
            K {float} -- the fixed ordering cost
            v {float} -- the per unit ordering cost
            h {float} -- the per unit holding cost
            d {List[float]} -- the demand in each period
            I0 {float} -- the initial inventory level
        """
        super().__init__(K, v, h, d, I0, C)

        # initialize instance variables
        self.T, min_inv, max_inv = len(d), 0, sum(d)

        # auxiliary parameters
        M = 100000 # a big number

        # lambdas
        self.ag = lambda s: [x for x in range(0, min(max_inv-s.I, self.C+1))] # action generator
        self.st = lambda s, a, d: State(s.t+1, s.I+a-d)                       # state transition
        L = lambda i,a,d : self.h*max(i+a-d, 0) + M*max(d-i-a, 0)             # immediate holding/penalty cost
        self.iv = lambda s, a, d: (self.K+v*a if a > 0 else 0) + L(s.I, a, d) # immediate value function

        self.cache_actions = {}                                               # cache with optimal state/action pairs

        print("Total cost: " + str(self.f(self.I0)))
        print("Order quantities: " + str([Q for Q in self.order_quantities()]))

    def f(self, level: float) -> float:
        """
        Recursively solve the capacitated lot sizing problem
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

        Arguments:
            period {int} -- the initial period
            level {float} -- the initial inventory level
        
        Returns:
            float -- the optimal order quantity 
        """

        s = State(period,level)
        if not(str(s) in self.cache_actions):
            self._f(s)
            
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
        v = min(                                                             # optimal cost
            [(self.iv(s, a, self.d[s.t])+                                    # immediate cost
             (self._f(self.st(s, a, self.d[s.t])) if s.t < self.T-1 else 0)) # future cost                            
             for a in self.ag(s)])                                           # actions

        opt_a = lambda a: (self.iv(s, a, self.d[s.t])+                       # optimal cost
                          (self._f(self.st(s, a, self.d[s.t])) if s.t < self.T-1 else 0)) == v          
                               
        q = [k for k in filter(opt_a, self.ag(s))]                              # retrieve best action list
        self.cache_actions[str(s)]=q[0] if bool(q) else None                    # store an action in dictionary
        return v                                                                # return expected total cost

    def _compute_order_quantities(self) -> List[float]:
        '''
        Compute optimal capacitated lot sizing order quantities
        '''
        I = self.I0  
        for t in range(len(self.d)):
            Q = self.q(t, I)
            I += Q - self.d[t]
            yield Q

    def order_quantities(self) -> List[float]:
        return [Q for Q in self._compute_order_quantities()]
    
    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the capacitated lot sizing problem
        '''
        return self.f(self.I0)

    @staticmethod
    def _test():
        print("********** CapacitatedLotSizingSDP **********")
        instance = {"K": 40, "v": 1, "h": 1, "d":[10,20,30,40], "I0": 0, "C": 30}
        CapacitatedLotSizingSDP(**instance)


if __name__ == '__main__':
    CapacitatedLotSizingCPLEX._test()
    CapacitatedLotSizingSDP._test()