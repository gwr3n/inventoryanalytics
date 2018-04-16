'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class eoq:
    '''
    Ford W. Harris, How Many Parts to Make at Once, Factory, 
    The Magazine of Management, Volume 10, Number 2, February 1913, pp. 135–136, 152.

    Harris, Ford W. (1990). "How Many Parts to Make at Once". Operations Research. 
    38 (6): 947. doi:10.1287/opre.38.6.947.
    '''

    def __init__(self, K: float, h: float, d: float, v: float):
        """
        Constructs an instance of the Economic Order Quantity problem.
        
        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the proportional holding cost
            d {float} -- the demand per period
            v {float} -- the unit purchasing cost
        """

        self.K, self.h, self.d, self.v = K, h, d, v

    def compute_eoq(self) -> float:
        """
        Computes the Economic Order Quantity.
        
        Returns:
            float -- the Economic Order Quantity
        """

        x0 = 1 # start from a positive EOQ
        res = minimize(self.relevant_cost, x0, method='nelder-mead', 
                       options={'xtol': 1e-8, 'disp': False})
        return res.x[0]

    def relevant_cost(self, Q: float) -> float:
        """
        Computes the relevant cost (ignoring the unit production cost) 
        per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.ch(Q)

    def cost(self, Q: float) -> float:
        """
        Computes the total cost per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.co_variable(Q)+self.ch(Q)

    def co_fixed(self, Q: float) -> float:
        """
        Computes the fixed ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the fixed ordering cost
        """

        K, d= self.K, self.d
        return K/(Q/d)

    def co_variable(self, Q: float) -> float:
        """
        Computes the variable ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the variable ordering cost
        """        
        d, v = self.d, self.v
        return d*v

    def ch(self, Q: float) -> float:
        """
        Computes the inventory holding cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the inventory holding cost
        """
        h = self.h
        return h*Q/2

    def coverage(self) -> float:
        """
        Compute the number of periods of demand the 
        Economic Order Quantity will satisfy 
        (i.e. the replenishment cycle length).
        
        Returns:
            float -- the number of periods of demand the 
                Economic Order Quantity will satisfy
        """

        d = self.d
        return self.compute_eoq()/d

    def average_inventory(self) -> float:
        """
        Computes the average inventory level 
        
        Returns:
            float -- the average inventory level 
        """

        return self.compute_eoq()/2

    def itr(self) -> float:
        """
        The Implied Turnover Ratio (ITR) represents the number of times 
        inventory is sold or used in a time period.
        
        Returns:
            float -- the Implied Turnover Ratio (ITR)
        """

        d = self.d
        return 2*d/self.compute_eoq()

    def sensitivity_to_Q(self, Q: float) -> float:
        """
        Computes the additional cost faced if the 
        chosen order quantity `Q` deviates from the 
        optimal order quantity.
        
        Arguments:
            Q {float} -- the target order quantity
        
        Returns:
            float -- a ratio indicating the percent 
                increase, e.g. 1.05 is a 5% increase
        """
        Qopt = self.compute_eoq()
        return 0.5*(Qopt/Q+Q/Qopt)

    def reorder_point(self, lead_time: float) -> float:
        """
        Computes the reorder point for a given lead time.
        
        Arguments:
            lead_time {float} -- the given lead time
        
        Returns:
            float -- the reorder point
        """

        d = self.d
        return d*lead_time
    
    def opt_powersoftwo_policy(self, T: float) -> float:
        K, d, h = self.K, self.d, self.h
        rc = lambda t : K/t + h*d*t/2
        k = 0
        while rc(T*2**(k+1)) < rc(T*2**k):
            k += 1
        return T*2**k

    def sensitivity_to_K(self, K: float) -> float:
        """
        Computes the additional cost faced if the 
        esstimated K deviates from the actual one.
        
        Arguments:
            K {float} -- the estimated K
        
        Returns:
            float -- a ratio indicating the percent 
                increase, e.g. 1.05 is a 5% increase
        """
        e = lambda x : x + 1/x
        return 0.5*(e(np.sqrt(K/self.K)))

    def sensitivity_to_h(self, h: float) -> float:
        """
        Computes the additional cost faced if the 
        esstimated h deviates from the actual one.
        
        Arguments:
            h {float} -- the estimated h
        
        Returns:
            float -- a ratio indicating the percent 
                increase, e.g. 1.05 is a 5% increase
        """
        e = lambda x : x + 1/x
        return 0.5*(e(np.sqrt(self.h/h)))

    @staticmethod
    def _plot_eoq():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2}
        pb = eoq(**instance)
        total, = plt.plot([k for k in range(10,100)], 
                          [pb.relevant_cost(k) for k in range(10,100)], 
                          label='Total relevant cost')
        ordering, = plt.plot([k for k in range(10,100)], 
                             [pb.co_fixed(k) for k in range(10,100)], 
                             label='Ordering cost')
        holding, = plt.plot([k for k in range(10,100)], 
                            [pb.ch(k) for k in range(10,100)], 
                            label='Holding cost')
        plt.legend(handles=[total,ordering,holding], loc=1)
        plt.ylabel('Cost')
        plt.xlabel('Q')
        plt.show()
    
    @staticmethod
    def _plot_sensitivity_to_Q():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2}
        pb = eoq(**instance)
        QOpt = int(np.round(pb.compute_eoq()))
        plt.plot([k for k in range(20-QOpt,80-QOpt)], [pb.sensitivity_to_Q(k) for k in range(20,80)])
        plt.ylabel('Sensitivity')
        plt.xlabel('Difference between Q and Q*')
        plt.show() 
    
    @staticmethod
    def _plot_sensitivity_to_K():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2}
        pb = eoq(**instance)
        plt.plot([k for k in range(60-pb.K,140-pb.K)], [pb.sensitivity_to_K(k) for k in range(60,140)])
        plt.ylabel('Sensitivity')
        plt.xlabel('Difference between K\' and K')
        plt.show() 

    @staticmethod
    def _plot_sensitivity_to_h():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2}
        pb = eoq(**instance)
        plt.plot([k*0.1-pb.h for k in range(1,30)], [pb.sensitivity_to_h(k*0.1) for k in range(1,30)])
        plt.ylabel('Sensitivity')
        plt.xlabel('Difference between h\' and h')
        plt.show() 

    @staticmethod
    def _sample_instance():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2}
        pb1 = eoq(**instance)
        Qopt = pb1.compute_eoq()
        print(Qopt)
        print(pb1.relevant_cost(Qopt))
        instance = {"K": 60, "h": 1, "d": 10, "v": 2}
        pb2 = eoq(**instance)
        Qopt = pb2.compute_eoq()
        print(pb1.relevant_cost(Qopt))

class eoq_all_units_discounts:
    pass

class eoq_incremental_discounts:
    pass

class eoq_planned_backorders:
    def __init__(self, K: float, h: float, d: float, v: float, p: float):
        """
        Constructs an instance of the Economic Order Quantity problem.
        
        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the proportional holding cost
            d {float} -- the demand per period
            v {float} -- the unit purchasing cost
            p {float} -- the backordering penalty cost
        """

        self.K, self.h, self.d, self.v, self.p = K, h, d, v, p

    def compute_eoq(self) -> float:
        """
        Computes the Economic Order Quantity.
        
        Returns:
            float -- the Economic Order Quantity
        """

        x0 = 1 # start from a positive EOQ
        res = minimize(self.relevant_cost, x0, method='nelder-mead', 
                       options={'xtol': 1e-8, 'disp': False})
        return res.x[0]

    def relevant_cost(self, Q: float) -> float:
        """
        Computes the relevant cost (ignoring the unit production cost) 
        per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.ch(Q)+self.cp(Q)

    def cost(self, Q: float) -> float:
        """
        Computes the total cost per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.co_variable(Q)+self.ch(Q)+self.cp(Q)

    def co_fixed(self, Q: float) -> float:
        """
        Computes the fixed ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the fixed ordering cost
        """

        K, d= self.K, self.d
        return K/(Q/d)

    def co_variable(self, Q: float) -> float:
        """
        Computes the variable ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the variable ordering cost
        """        
        d, v = self.d, self.v
        return d*v

    def ch(self, Q: float) -> float:
        """
        Computes the inventory holding cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the inventory holding cost
        """
        h = self.h
        x = self.h/(self.p+self.h)
        return h*(Q-Q*x)**2/(2*Q)
    
    def cp(self, Q: float) -> float:
        """
        Computes the inventory penalty cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the inventory penalty cost
        """
        p = self.p
        x = self.h/(self.p+self.h)
        return p*(Q*x)**2/(2*Q)

    @staticmethod
    def _sample_instance():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2, "p": 5}
        #instance = {"K": 8, "h": 0.3*0.75, "d": 1300, "v": 75, "p": 5}
        pb = eoq_planned_backorders(**instance)
        Qopt = pb.compute_eoq()
        print(Qopt)
        print(np.sqrt(2*pb.K*pb.d*(pb.h+pb.p)/(pb.h*pb.p)))
        print(pb.h/(pb.p+pb.h))
        print(pb.relevant_cost(Qopt))

class epq:
    def __init__(self, K: float, h: float, d: float, v: float, p: float):
        """
        Constructs an instance of the Economic Order Quantity problem.
        
        Arguments:
            K {float} -- the fixed ordering cost
            h {float} -- the proportional holding cost
            d {float} -- the demand per period
            v {float} -- the unit purchasing cost
            p {float} -- the finite production rate
        """

        self.K, self.h, self.d, self.v, self.p = K, h, d, v, p

    def compute_eoq(self) -> float:
        """
        Computes the Economic Order Quantity.
        
        Returns:
            float -- the Economic Order Quantity
        """

        x0 = 1 # start from a positive EOQ
        res = minimize(self.relevant_cost, x0, method='nelder-mead', 
                       options={'xtol': 1e-8, 'disp': False})
        return res.x[0]

    def relevant_cost(self, Q: float) -> float:
        """
        Computes the relevant cost (ignoring the unit production cost) 
        per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.ch(Q)

    def cost(self, Q: float) -> float:
        """
        Computes the total cost per unit period for a given quantity Q.
        
        Arguments:
            Q {float} -- the order quantity

        Returns:
            float -- the optimal cost per unit period
        """

        return self.co_fixed(Q)+self.co_variable(Q)+self.ch(Q)

    def co_fixed(self, Q: float) -> float:
        """
        Computes the fixed ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the fixed ordering cost
        """

        K, d= self.K, self.d
        return K/(Q/d)

    def co_variable(self, Q: float) -> float:
        """
        Computes the variable ordering cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the variable ordering cost
        """        
        d, v = self.d, self.v
        return d*v

    def ch(self, Q: float) -> float:
        """
        Computes the inventory holding cost
        
        Arguments:
            Q {float} -- the order quantity
        
        Returns:
            float -- the inventory holding cost
        """
        h = self.h
        rho = self.p/self.d
        return h*Q*(1-rho)/2

    @staticmethod
    def _sample_instance():
        instance = {"K": 100, "h": 1, "d": 10, "v": 2, "p": 5}
        #instance = {"K": 8, "h": 0.3*0.75, "d": 1300, "v": 75, "p": 5}
        pb = epq(**instance)
        Qopt = pb.compute_eoq()
        print(Qopt)
        rho = pb.p/pb.d
        print(np.sqrt(2*pb.K*pb.d/(pb.h*(1-rho))))
        print(np.sqrt(2*pb.K*pb.d*(pb.h*(1-rho))))

if __name__ == '__main__':
    #eoq._plot_eoq()
    #eoq._plot_sensitivity_to_Q()
    #eoq._plot_sensitivity_to_K()
    #eoq._plot_sensitivity_to_h()
    #eoq._sample_instance()
    #eoq_planned_backorders._sample_instance()
    epq._sample_instance()