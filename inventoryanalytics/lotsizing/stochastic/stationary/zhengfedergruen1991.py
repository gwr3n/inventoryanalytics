'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np
import functools 
from scipy.stats import poisson
from inventoryanalytics.utils import memoize as mem

def expectation(f, x, p): # E f(X) = sum f(x_i) p_i
    return np.dot(f(x),p)

class ZhengFedergruen(object):
    """[summary]
    The stationary stochastic lot sizing problem.

    Zheng, Y., & Federgruen, A. (1991). Finding optimal (s, s) policies is about
    as simple as evaluating a single policy. Operations research, 39 , 654{665.
    doi:10.1287/opre.39.4.654.
    
    """

    def __init__(self, mu, K, h, b):
        """
        Constructs an instance of the stochastic lot sizing problem
        
        Arguments:
            mu {[type]} -- the expected demand
            K {[type]} -- the fixed ordering cost
            h {[type]} -- the proportional holding cost
            b {[type]} -- the penalty cost
        """

        self.K = K      
        self.mu = mu
        self.p = poisson.pmf(np.arange(200), self.mu)
        self.h = h
        self.b = b

    def G_L(self, y): # the one-period inventory cost
        # see Page 655
        return self.b*np.maximum(0,-y) + self.h*np.maximum(0,y)

    def G(self, y): # expected one period inventory cost
        return expectation(self.G_L, y - np.arange(0, len(self.p)), self.p)

    @mem.memoize
    def m(self, j):  # 2a
        if j == 0:
            return 1./(1. - self.p[0])
        else:        # 2b
            res = sum(self.p[l]*self.m(j-l) for l in range(1,j+1))
            res /= (1. - self.p[0])
            return res

    @mem.memoize
    def M(self, j):  # 2c
        if j == 0:
            return 0.
        else:
            return self.M(j-1) + self.m(j-1)

    def k(self, s,y): # 5
        res = self.K
        res += sum(self.m(j)*self.G(y-j) for j in range(y-s))
        return res

    def c(self, s,S): # 3
        return self.k(s,S)/self.M(S-s)


    def findOptimalPolicy(self):
        
        # algorithm on Page 659
        ystar = poisson.ppf(self.b/(self.b+self.h),self.mu).astype(int) #base stock level
        s = ystar - 1   #upper bound for s
        S_0 = ystar + 0 #lower bound for S_0

        #calculate the optimal s for S fixed at its lower bound S0
        while self.c(s,S_0) > self.G(s):
            s -= 1
        s_0 = s # + 0   #optimal value of s for S0
        c0 = self.c(s_0,S_0) #costs for this starting value
        S0 = S_0 # + 0  # S0 = S^0 of the paper
        S = S0+1
        while self.G(S) <= c0:
            if self.c(s,S) < c0:
                S0 = S+0
                while self.c(s,S0) <= self.G(s+1):
                    s += 1
                c0 = self.c(s,S0)
            S += 1
            #print(str(s) + " " + str(S))
        self.s_star = s
        self.S_star = S0
        return s, S0