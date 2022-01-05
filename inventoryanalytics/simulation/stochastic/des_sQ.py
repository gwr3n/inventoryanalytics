import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from queue import PriorityQueue
from collections import defaultdict
from scipy.optimize import minimize

def plot_inventory(values, label):

    # data 
    df=pd.DataFrame({'x': np.array(values)[:,0], 'fx': np.array(values)[:,1]})
    
    # plot
    plt.xticks(range(len(values)),
               range(1,len(values)+1))
    plt.xlabel("$t$")
    plt.ylabel("items")
    plt.plot( 'x', 'fx', data=df, linestyle='-', marker='', label=label)

#########################
##         DES         ##
#########################

class EventWrapper():
    def __init__(self, event):
        self.event = event

    def __lt__(self, other):
        return self.event.priority < other.event.priority

class DES():
    def __init__(self, end):
        self.events, self.end, self.time = PriorityQueue() , end, 0
    
    def start(self):
        while True:
            event = self.events.get()
            self.time = event[0]
            if self.time <= self.end:
                event[1].event.end()
            else:
                break

    def schedule(self, event: EventWrapper, time_lag: int):
        self.events.put((self.time + time_lag, event))

##########################
##       WAREHOUSE      ##
##########################

class Warehouse:
    def __init__(self, inventory_level, fixed_ordering_cost, holding_cost, penalty_cost):
        self.i, self.K, self.h, self.p = inventory_level, fixed_ordering_cost, holding_cost, penalty_cost
        self.o = 0 # outstanding_orders
        self.period_costs = defaultdict(int) # a dictionary recording cost in each period

    def receive_order(self, Q, time):
        self.review_inventory(time)
        self.i, self.o = self.i + Q, self.o - Q
        self.review_inventory(time)

    def order(self, Q, time):
        self.review_inventory(time)
        self.period_costs[time] += self.K # incur ordering cost and store it in a dictionary
        self.o += Q
        self.review_inventory(time) 
    
    def on_hand_inventory(self):
        return max(0,self.i)
    
    def backorders(self):
        return max(0,-self.i)

    def issue(self, demand, time):
        self.review_inventory(time)
        self.i = self.i-demand

    def inventory_position(self):
        return self.o+self.i

    def review_inventory(self, time):
        try:
            self.levels.append([time, self.i]) 
            self.on_hand.append([time, self.on_hand_inventory()]) 
            self.positions.append([time, self.inventory_position()]) 
        except AttributeError:
            self.levels, self.on_hand = [[0, self.i]], [[0, self.on_hand_inventory()]]
            self.positions = [[0, self.inventory_position()]]
    
    def incur_end_of_period_costs(self, time): # incur holding and penalty costs
        self._incur_holding_cost(time)
        self._incur_penalty_cost(time)

    def _incur_holding_cost(self, time): # incur holding cost and store it in a dictionary
        self.period_costs[time] += self.on_hand_inventory()*self.h
    
    def _incur_penalty_cost(self, time): # incur penalty cost and store it in a dictionary
        self.period_costs[time] += self.backorders()*self.p

##########################
##         EVENTS       ##
##########################

class CustomerDemand:
    def __init__(self, des: DES, demand_rate: float, warehouse: Warehouse):
        self.d = demand_rate # the demand rate per period
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.priority = 2 # denotes a low priority
    
    def end(self):
        self.w.issue(1, self.des.time)
        self.des.schedule(EventWrapper(self), np.random.exponential(1/self.d)) # schedule another demand

class EndOfPeriod:
    def __init__(self, des: DES, warehouse: Warehouse):
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.priority = 0 # denotes a low priority
    
    def end(self):
        self.w.incur_end_of_period_costs(self.des.time-1)
        self.des.schedule(EventWrapper(EndOfPeriod(self.des, self.w)), 1)

class InventoryReview:
    def __init__(self, des: DES, s: float, Q: float, warehouse: Warehouse, lead_time: float):
        self.s, self.Q = s, Q # the reorder point and the order quantity
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.lead_time = lead_time
        self.priority = 1 # denotes a medium priority

    def end(self):
        if self.w.inventory_position() < self.s:
            self.w.order(self.Q, self.des.time)
            self.des.schedule(EventWrapper(ReceiveOrder(self.des, self.Q, self.w)), self.lead_time)
        self.des.schedule(EventWrapper(self), 1) # schedule another review in 1 period
        
class ReceiveOrder:
    def __init__(self, des: DES, Q: float, warehouse: Warehouse):
        self.Q = Q # the order quantity
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.priority = 1 # denotes a medium priority
    
    def end(self):
        self.w.receive_order(self.Q, self.des.time)

class sQ:
    def __init__(self, instance, demand, lead_time, N):
        self.instance, self.demand, self.lead_time, self.N = instance, demand, lead_time, N

    def _run_DES(self, parameters):
        s, Q = tuple(parameters)
        np.random.seed(1234)
        w = Warehouse(**self.instance)

        des = DES(self.N)
        d = CustomerDemand(des, self.demand, w)
        des.schedule(EventWrapper(d), 0) # schedule a demand immediately

        o = InventoryReview(des, s, Q, w, self.lead_time)
        des.schedule(EventWrapper(o), 0) # schedule an order immediately
        des.schedule(EventWrapper(EndOfPeriod(des, w)), 1) # schedule EndOfPeriod at the end of the first period
        des.start()
        return w

    def simulate(self, parameters):
        w = self._run_DES(parameters)

        return (sum([w.period_costs[e] for e in w.period_costs])/len(w.period_costs))
    
    def plot(self, parameters):
        w = self._run_DES(parameters)
        
        print("Period costs: "+str([w.period_costs[e] for e in w.period_costs]))
        print("Average cost per period: "+ '%.2f' % (sum([w.period_costs[e] for e in w.period_costs])/len(w.period_costs)))
        plot_inventory(w.positions, "inventory position")
        plot_inventory(w.levels, "inventory level")
        s = parameters[0]
        plt.plot([s for k in range(self.N)], label="s")
        plt.legend(loc="lower right")
        plt.savefig('/Users/gwren/Downloads/14_sQ_policy.svg', format='svg')
        plt.show()

    def plot_surface(self, sq, execution_path):
        x = range(0,20)
        y = range(25,55)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[sq.simulate([s,Q]) for s in x] for Q in y])
        plt.contourf(X, Y, Z, levels=100, cmap ="bone")
        plt.plot(np.array(execution_path)[:,0],np.array(execution_path)[:,1],'ro--',linewidth=0.5, markersize=1) 
        plt.colorbar()
        plt.xlabel("s")
        plt.ylabel("Q")
        plt.savefig('/Users/gwren/Downloads/15_nelder_mead_sQ.svg', format='svg')
        plt.show()

def plot_sQ():
    instance = {"inventory_level": 0, "fixed_ordering_cost": 64, "holding_cost": 1, "penalty_cost": 9}
    demand, lead_time, s, Q = 10, 0, 6, 39
    N = 20
    sq = sQ(instance, demand, lead_time, N)
    sq.plot([s, Q])

def optimize_sQ(surface):
    instance = {"inventory_level": 0, "fixed_ordering_cost": 64, "holding_cost": 1, "penalty_cost": 9}
    demand, lead_time, s, Q = 10, 0, 6, 40
    N = 1000 # planning horizon length
    sq = sQ(instance, demand, lead_time, N)

    execution_path = []
    def callbackF(x):
        print('{0}\t{1}\t{2} '.format(x[0], x[1], sq.simulate(x)))
        execution_path.append([x[0], x[1]])

    #methods = ["Powell", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
    m = "Nelder-Mead"
    res = minimize(sq.simulate, [s,Q], method=m, callback=callbackF, options={"maxiter": 50})
    print([m, list(res.x), res.fun])

    if surface:
        sq.plot_surface(sq, execution_path)

# plot_sQ()
plot_surface = True
optimize_sQ(plot_surface)
