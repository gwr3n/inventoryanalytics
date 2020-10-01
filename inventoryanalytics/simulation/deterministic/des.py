import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from queue import PriorityQueue
from collections import defaultdict

def plot_inventory(values, label):

    # data 
    df=pd.DataFrame({'x': np.array(values)[:,0], 'fx': np.array(values)[:,1]})
    
    # plot
    plt.xticks(range(len(values)),
               range(1,len(values)+1))
    plt.xlabel("t")
    plt.ylabel("items")
    plt.plot( 'x', 'fx', data=df, linestyle='-', marker='o', label=label)

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
            if self.time < self.end:
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
        self.priority = 1 # denotes a low priority
    
    def end(self):
        self.w.issue(self.d, self.des.time)
        self.des.schedule(EventWrapper(self), 1) # schedule another demand

class EndOfPeriod:
    def __init__(self, des: DES, warehouse: Warehouse):
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.priority = 2 # denotes a low priority
    
    def end(self):
        self.w.incur_end_of_period_costs(self.des.time)
        self.des.schedule(EventWrapper(EndOfPeriod(self.des, self.w)), 1)

class Order:
    def __init__(self, des: DES, Q: float, warehouse: Warehouse, lead_time: float):
        self.Q = Q # the order quantity
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.lead_time = lead_time
        self.priority = 0 # denotes a high priority

    def end(self):
        self.w.order(self.Q, self.des.time)
        self.des.schedule(EventWrapper(ReceiveOrder(self.des, self.Q, self.w)), self.lead_time)
        
class ReceiveOrder:
    def __init__(self, des: DES, Q: float, warehouse: Warehouse):
        self.Q = Q # the order quantity
        self.w = warehouse # the warehouse
        self.des = des # the Discrete Event Simulation engine
        self.priority = 0 # denotes a high priority
    
    def end(self):
        self.w.receive_order(self.Q, self.des.time)

np.random.seed(1234)

instance = {"inventory_level": 0, "fixed_ordering_cost": 100, "holding_cost": 1, "penalty_cost": 5}
w = Warehouse(**instance)

N = 20 # planning horizon length
des = DES(N)

d = CustomerDemand(des, 10, w)
des.schedule(EventWrapper(d), 0) # schedule a demand immediately

lead_time = 0
o = Order(des, 50, w, lead_time)
for t in range(0,20,5):
    des.schedule(EventWrapper(o), t) # schedule orders
des.schedule(EventWrapper(EndOfPeriod(des, w)), 0) # schedule EndOfPeriod immediately

des.start()

print("Period costs: "+str([w.period_costs[e] for e in w.period_costs]))
print("Average cost per period: "+ '%.2f' % (sum([w.period_costs[e] for e in w.period_costs])/len(w.period_costs)))

plot_inventory(w.positions, "inventory position")
plot_inventory(w.levels, "inventory level")
plt.legend(loc="lower right")
plt.show()