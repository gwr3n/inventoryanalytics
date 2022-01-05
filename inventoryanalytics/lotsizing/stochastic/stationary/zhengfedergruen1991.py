'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np
from scipy.stats import poisson
from inventoryanalytics.utils import memoize as mem

import matplotlib.pyplot as plt

def expectation(f, x, p): # E f(X) = sum f(x_i) p_i
    return np.dot(f(x),p)

class ZhengFedergruen(object):
    """
    The stationary stochastic lot sizing problem.

    Zheng, Y., & Federgruen, A. (1991). Finding optimal (s, s) policies is about
    as simple as evaluating a single policy. Operations research, 39 , 654-665.
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
        self.p = poisson.pmf(np.arange(10*mu), self.mu)
        self.h = h
        self.b = b
        self.execution_path = []

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
        self.execution_path.append([s,S_0])
        while self.c(s,S_0) > self.G(s):
            s -= 1
            self.execution_path.append([s,S_0])
        s_0 = s # + 0   #optimal value of s for S0
        c0 = self.c(s_0,S_0) #costs for this starting value
        S0 = S_0 # + 0  # S0 = S^0 of the paper
        S = S0+1
        self.execution_path.append([s,S])
        while self.G(S) <= c0:
            if self.c(s,S) < c0:
                S0 = S+0
                while self.c(s,S0) <= self.G(s+1):
                    s += 1
                    self.execution_path.append([s,S0])
                c0 = self.c(s,S0)
                self.execution_path.append([s,S])
            S += 1
        #print(np.array(self.execution_path))
        #self.plot()
        self.s_star = s
        self.S_star = S0
        return s, S0

    def plot(self):
        plt.plot(np.array(self.execution_path)[:,0],np.array(self.execution_path)[:,1],'b-')
        plt.arrow(np.array(self.execution_path)[0,0],np.array(self.execution_path)[0,1],
                           0.1*(np.array(self.execution_path)[1,0]-np.array(self.execution_path)[0,0]),
                           0.1*(np.array(self.execution_path)[1,1]-np.array(self.execution_path)[0,1]), 
                           lw=0.1, length_includes_head=False, head_width=0.7, head_length=0.3)
        plt.arrow(np.array(self.execution_path)[10,0],np.array(self.execution_path)[10,1],
                           0.1*(np.array(self.execution_path)[11,0]-np.array(self.execution_path)[10,0]),
                           0.1*(np.array(self.execution_path)[11,1]-np.array(self.execution_path)[10,1]), 
                           lw=0.5, length_includes_head=False, head_width=0.3, head_length=0.5)
        plt.arrow(np.array(self.execution_path)[20,0],np.array(self.execution_path)[20,1],
                           0.1*(np.array(self.execution_path)[21,0]-np.array(self.execution_path)[20,0]),
                           0.1*(np.array(self.execution_path)[21,1]-np.array(self.execution_path)[20,1]), 
                           lw=0.5, length_includes_head=False, head_width=0.3, head_length=0.5)
        plt.arrow(np.array(self.execution_path)[30,0],np.array(self.execution_path)[30,1],
                           0.1*(np.array(self.execution_path)[31,0]-np.array(self.execution_path)[30,0]),
                           0.1*(np.array(self.execution_path)[31,1]-np.array(self.execution_path)[30,1]), 
                           lw=0.5, length_includes_head=False, head_width=0.3, head_length=0.5)
        plt.xlabel("s")
        plt.ylabel("S")
        plt.savefig('/Users/gwren/Downloads/10_zf_execution_path.eps', format='eps')
        plt.show()

    @staticmethod
    def run_instance():
        instance = {'mu': 10, 'K': 64, 'h': 1., 'b': 9.}
        pb = ZhengFedergruen(**instance)
        print(pb.findOptimalPolicy())
    