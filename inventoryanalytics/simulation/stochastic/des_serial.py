import matplotlib.pyplot as plt, numpy as np, pandas as pd
from queue import PriorityQueue
from collections import defaultdict
from typing import List

def plot_inventory(values_warehouse, values_retailer):
    fig, axs = plt.subplots(2)

    # data 
    dfW=pd.DataFrame({'x': np.array(values_warehouse)[:,0], 'fx': np.array(values_warehouse)[:,1]})
    dfR=pd.DataFrame({'x': np.array(values_retailer)[:,0], 'fx': np.array(values_retailer)[:,1]})
    
    # plot
    axs[0].set_xticks(range(len(values_warehouse)), range(1,len(values_warehouse)+1))
    axs[0].set_xlabel("t")
    #axs[0].ylabel("items")
    axs[0].plot( 'x', 'fx', data=dfW, linestyle='-', marker='', label="W")
    axs[0].set_title("Warehouse")

    # plot
    axs[1].set_xticks(range(len(values_retailer)), range(1,len(values_retailer)+1))
    axs[1].set_xlabel("t")
    #axs[1].ylabel("items")
    axs[1].plot( 'x', 'fx', data=dfR, linestyle='-', marker='', label="R")
    axs[1].set_title("Retailer")
    fig.tight_layout()
    fig.show()

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
##    Installations     ##
##########################

class Warehouse:
    def __init__(self, inventory_level, holding_cost, lead_time):
        self.i, self.h, self.lead_time = inventory_level, holding_cost, lead_time
        self.o = 0 # outstanding_orders
        self.period_costs = defaultdict(int) # a dictionary recording cost in each period

    def receive_order(self, Q, time):
        self.review_inventory(time)
        self.i, self.o = self.i + Q, self.o - Q
        self.review_inventory(time)

    def order(self, Q, time):
        self.review_inventory(time)
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

    def _incur_holding_cost(self, time): # incur holding cost and store it in a dictionary
        self.period_costs[time] += self.on_hand_inventory()*self.h

class Retailer:
    def __init__(self, inventory_level, holding_cost, penalty_cost, lead_time, demand_rate):
        self.i, self.h, self.p, self.lead_time, self.demand_rate = inventory_level, holding_cost, penalty_cost, lead_time, demand_rate
        self.o = 0 # outstanding_orders
        self.period_costs = defaultdict(int) # a dictionary recording cost in each period

    def receive_order(self, Q, time):
        self.review_inventory(time)
        self.i, self.o = self.i + Q, self.o - Q
        self.review_inventory(time)

    def order(self, Q, time):
        self.review_inventory(time)
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
# Order of events
#
# 1. installation W orders;
# 2. the period delivery from the outside supplier S arrives at W;
# 3. installation R orders from installation W;
# 4. the period delivery from installation W arrives at installation R;
# 5. the stochastic customer demand at installation R is realised;
# 6. evaluation of holding and shortage costs.

class CustomerDemand:
    def __init__(self, des: DES, demand_rate: float, retailer: Retailer):
        self.d = demand_rate # the demand rate per period
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 5 # denotes a low priority
    
    def end(self):
        self.r.issue(1, self.des.time)
        self.des.schedule(EventWrapper(self), np.random.exponential(1/self.d)) # schedule another demand

class EndOfPeriod:
    def __init__(self, des: DES, warehouse: Warehouse, retailer: Retailer):
        self.w = warehouse # the warehouse
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 0 # denotes a high priority
    
    def end(self):
        self.w.incur_end_of_period_costs(self.des.time-1)
        self.r.incur_end_of_period_costs(self.des.time-1)
        self.des.schedule(EventWrapper(EndOfPeriod(self.des, self.w, self.r)), 1)

class OrderUpTo_Warehouse:
    def __init__(self, des: DES, S: float, warehouse: Warehouse, retailer: Retailer):
        self.S = S # the order-up-to-position
        self.w = warehouse # the warehouse
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 1 # denotes a medium priority

    def end(self):
        Q = self.S - self.w.inventory_position()
        self.w.order(Q, self.des.time)
        self.des.schedule(EventWrapper(ReceiveOrder_Warehouse(self.des, Q, self.w, self.r)), self.w.lead_time)

class OrderUpTo_Retailer:
    def __init__(self, des: DES, S: float, warehouse: Warehouse, retailer: Retailer):
        self.S = S # the order-up-to-position
        self.w = warehouse # the warehouse
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 2 # denotes a medium priority

    def end(self):
        Q = self.S - self.r.inventory_position()
        self.r.order(Q, self.des.time)
        Q_available = min(Q, self.w.on_hand_inventory())
        self.w.issue(Q, self.des.time)
        self.des.schedule(EventWrapper(ReceiveOrder_Retailer(self.des, Q_available, self.r)), self.r.lead_time)
        
class ReceiveOrder_Warehouse:
    def __init__(self, des: DES, Q: float, warehouse: Warehouse, retailer: Retailer):
        self.Q = Q # the order quantity
        self.w = warehouse # the warehouse
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 3 # denotes a medium priority
    
    def end(self):
        backorders = self.w.backorders()
        self.w.receive_order(self.Q, self.des.time)
        if backorders > 0:
            q = min(self.Q, backorders)
            self.des.schedule(EventWrapper(ReceiveOrder_Retailer(self.des, q, self.r)), self.r.lead_time)

class ReceiveOrder_Retailer:
    def __init__(self, des: DES, Q: float, retailer: Retailer):
        self.Q = Q # the order quantity
        self.r = retailer # the retailer
        self.des = des # the Discrete Event Simulation engine
        self.priority = 4 # denotes a medium priority
    
    def end(self):
        self.r.receive_order(self.Q, self.des.time)

def simulate(retailer, S_r, warehouse, S_w, N, plot):
    np.random.seed(1234)
    r, w = Retailer(**retailer), Warehouse(**warehouse)

    des = DES(N)
    d = CustomerDemand(des, r.demand_rate, r)
    des.schedule(EventWrapper(d), 0) # schedule a demand immediately

    o_r = OrderUpTo_Retailer(des, S_r, w, r)
    o_w = OrderUpTo_Warehouse(des, S_w, w, r)
    for t in range(N):
        des.schedule(EventWrapper(o_r), t) # schedule orders
        des.schedule(EventWrapper(o_w), t) # schedule orders
    des.schedule(EventWrapper(EndOfPeriod(des, w, r)), 1) # schedule EndOfPeriod at the end of the first period
    des.start()

    if plot:
        cW, cR = 0, 0
        while w.levels[cW][0] < 5:
            cW += 1
        while r.levels[cR][0] < 5:
            cR += 1
        plot_inventory(w.levels[cW:], r.levels[cR:])
        plt.savefig('/Users/gwren/Downloads/3_serial_inventory_level.eps', format='eps')
        plt.show()
    #print("Warehouse Period costs: "+str([w.period_costs[e] for e in w.period_costs]))
    #print("Warehouse inventory level: "+str(w.levels))
    #print("Retailer Period costs: "+str([r.period_costs[e] for e in r.period_costs]))
    #print("Retailer inventory level: "+str(r.levels))
    tc = sum([w.period_costs[e] for e in w.period_costs]) + sum([r.period_costs[e] for e in r.period_costs])
    return tc/N

def plot_surface(retailer, warehouse, N):
    S_r = range(70,80)
    S_w = range(55,65)
    X, Y = np.meshgrid(S_r, S_w)
    Z = np.array([[simulate(
        {"inventory_level": s_r, "holding_cost": retailer["holding_cost"], "penalty_cost": retailer["penalty_cost"], "lead_time": retailer["lead_time"], "demand_rate": retailer["demand_rate"]}, 
        s_r,  
        {"inventory_level": s_w, "holding_cost": warehouse["holding_cost"], "lead_time": warehouse["lead_time"]}, 
        s_w, N, False) for s_r in S_r] for s_w in S_w])
    plt.contourf(X, Y, Z, levels=20, cmap ="bone")
    plt.colorbar()
    plt.xlabel("S_r")
    plt.ylabel("S_w")
    plt.savefig('/Users/gwren/Downloads/4_serial_system_brute_force.svg', format='svg')
    plt.show()

S_r = 74
S_w = 59
retailer = {"inventory_level": S_r, "holding_cost": 1.5, "penalty_cost": 10, "lead_time": 5, "demand_rate": 10}
warehouse = {"inventory_level": S_w, "holding_cost": 1, "lead_time": 5}
N = 1000 # planning horizon length
#print("Average cost per period: "+ '%.2f' % simulate(retailer, S_r, warehouse, S_w, N, True))
plot_surface(retailer, warehouse, N)
