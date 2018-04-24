'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

from typing import List
import networkx as nx
import itertools

class WagnerWhitin:
    '''
    A Wagner-Whitin problem.

    H.M. Wagner and T. Whitin, 
    "Dynamic version of the economic lot size model," 
    Management Science, Vol. 5, pp. 89–96, 1958
    '''

    def __init__(self, K: float, h: float, d: List[float]):
        '''
        Create an instance of a Wagner-Whitin problem.

        K: the fixed ordering cost;
        h: the per unit holding cost;
        d: the demand in each period.
        '''
        self.K, self.h, self.d = K, h, d
        self.graph = nx.DiGraph()
        for i in range(0, len(self.d)):
            for j in range(i, len(self.d)):
                self.graph.add_edge(i, j, weight=self.cycle_cost(i, j))

    def cycle_cost(self, i: int, j: int) -> float:
        '''
        Compute the cost of a replenishment cycle covering periods i,...,j
        '''
        return self.K + \
               self.h * sum([(k-i)*self.d[k] for k in range(i,j+1)]) \
               if i<=j else 0

    def optimal_cost(self) -> float:
        '''
        Compute the cost of an optimal solution to the Wagner-Whitin problem
        '''
        T, cost, g = len(self.d), 0, self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        for t in range(1,len(path)):
            cost += self.cycle_cost(path[t-1],path[t]-1)
            print(self.cycle_cost(path[t-1],path[t]-1))
        return cost

    def order_quantities(self) -> List[float]:
        '''
        Compute optimal Wagner-Whitin order quantities
        '''
        T, g = len(self.d), self.graph
        path = nx.dijkstra_path(g, 0, T-1)
        path.append(len(self.d))
        qty = [0 for k in range(0,T)]
        for t in range(1,len(path)):
            qty[path[t-1]] = sum([self.d[k] for k in range(path[t-1],path[t])])
        return qty

if __name__ == '__main__':
    instance = {"K": 30, "h": 1, "d":[10,20,30,40]}
    ww = WagnerWhitin(**instance)
    print("Cost of an optimal plan: ", ww.optimal_cost())
    print("Optimal order quantities: ", ww.order_quantities())