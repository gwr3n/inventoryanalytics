'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''
from typing import List
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

class els:
    '''
    Jack Rogers. 
    A computational approach to the economic lot scheduling problem. 
    Management Science, 4(3): 264-291, April 1958
    '''

    def __init__(self, n: int, p: List[float], d: List[float], h: List[float], s: List[float], K: List[float]):
        """
        Constructs an instance of the Economic Lot Scheduling problem.

        Args:
            n (int): number of items
            p (List[float]): constant production rates
            d (List[float]): constant demand rates
            h (List[float]): inventory holding costs
            s (List[float]): set up times
            K (List[float]): set up costs
        """
        self.n, self.p, self.d, self.h, self.s, self.K = n, p, d, h, s, K

    def item_relevant_cost(self, T: float, i: int) -> float:
        """
        Computes the total relevant cost per unit period for item i and a given cycle length T.

        Args:
            T (float): the cycle length
            i (int): item index

        Returns:
            float: [description]
        """
        return 0.5*(self.h[i]*T*self.d[i]*(1-self.d[i]/self.p[i]))+self.K[i]/T

    def relevant_cost(self, T: float) -> float:
        """
        Computes the total relevant cost per unit period for a given cycle length T.

        Args:
            T (float): the cycle length

        Returns:
            float: the total relevant cost per unit period
        """
        return sum([self.item_relevant_cost(T,i) for i in range(0,self.n)])

    def compute_els(self):
        """
        Computes the Economic Lot Schedule.
        
        Returns:
            float -- the Economic Lot Schedule
        """
        return self._compute_els_nelder_mead()

    def _compute_els_nelder_mead(self):
        T0 = 1 # start from a positive T
        res = minimize(self.relevant_cost, T0, method='nelder-mead', 
                       options={'xtol': 1e-8, 'disp': False})
        return res.x[0]

    def _compute_els_closed_form(self):
        K = sum(self.K)
        H = sum([(self.h[i]*self.d[i]*(self.p[i]-self.d[i]))/(2*self.p[i]) for i in range(0,self.n)])
        return math.sqrt(K/H)
    
    def compute_production_cycle_length(self, T:float, i: int):
        return T*self.d[i]/self.p[i]

    def compute_max_inventory(self, T:float, i: int):
        return T*self.d[i]*(self.p[i]-self.d[i])/self.p[i]

    @staticmethod
    def _plot_els():
        instance = {"n": 3, "p": [400,400,500], "d": [50,50,60], "h": [20,20,30], "s":[0.1,0.1,0.1], "K": [2000,2500,800]}
        pb = els(**instance)
        total, = plt.plot([k for k in range(80,320)], 
                          [pb.relevant_cost(k/100.0) for k in range(80,320)], 
                          label='Total relevant cost')
        item1, = plt.plot([k for k in range(80,320)], 
                          [pb.item_relevant_cost(k/100.0,0) for k in range(80,320)], 
                          label='Total relevant cost item 1')
        item2, = plt.plot([k for k in range(80,320)], 
                          [pb.item_relevant_cost(k/100.0,1) for k in range(80,320)], 
                          label='Total relevant cost item 2')
        item3, = plt.plot([k for k in range(80,320)], 
                          [pb.item_relevant_cost(k/100.0,2) for k in range(80,320)], 
                          label='Total relevant cost item 3')
        plt.legend(handles=[total,item1,item2,item3], loc=1)
        plt.ylabel('Cost')
        plt.xlabel('$T$')
        x = [100,150,200,250,300]
        xNew = [1,1.5,2,2.5,3]
        plt.xticks(x, xNew)
        plt.savefig('/Users/gwren/Downloads/18_els_cost_plot.eps', format='eps')
        plt.show()

    @staticmethod
    def _sample_instance():
        instance = {"n": 3, "p": [400,400,500], "d": [50,50,60], "h": [20,20,30], "s":[0.1,0.1,0.1], "K": [2000,2500,800]}
        pb = els(**instance)
        Topt = pb.compute_els()
        print("ELS (nelder mead) = "+str(Topt))
        print("ELS (closed form) = "+str(pb._compute_els_closed_form()))
        for i in range(0,pb.n):
            print("Qmax["+str(i)+"] = "+str(pb.compute_max_inventory(Topt,i)))
            print("T["+str(i)+"] = "+str(pb.compute_production_cycle_length(Topt,i)))
        
if __name__ == '__main__':
    els._plot_els()
    #els._sample_instance()