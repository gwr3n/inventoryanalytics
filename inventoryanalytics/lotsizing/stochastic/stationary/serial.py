import math, matplotlib.pyplot as plt
from scipy.stats import poisson

def C_R(y, e_R, h_R, b, L_R, demand_rate):
    M = round(6*math.sqrt((L_R+1)*demand_rate)+(L_R+1)*demand_rate)
    return y*e_R-h_R*(L_R+1)*demand_rate+(h_R+b)*sum([(d-y)*poisson.pmf(d, (L_R+1)*demand_rate) for d in range(y,M)])

def compute_y_R(h_W, h_R, b, L_R, demand_rate):
    return poisson.ppf((h_W + b)/(h_R + b), (L_R+1)*demand_rate)

def C(y, e_R, h_R, b, L_R, h_W, L_W, demand_rate):    
    y_R = int(compute_y_R(h_W, h_R, b, L_R, demand_rate))
    CW = h_W*(y - (L_W+1)*demand_rate)
    CR = C_R(y_R, e_R, h_R, b, L_R, demand_rate)
    M = round(6*math.sqrt((L_W+1)*demand_rate)+(L_W+1)*demand_rate)
    s = sum([(C_R(y-d, e_R, h_R, b, L_R, demand_rate) - CR)*poisson.pmf(d, (L_W+1)*demand_rate) for d in range(y-y_R,M)])
    return CW + CR + s

def compute_y_W(e_R, h_R, b, L_R, h_W, L_W, demand_rate, initial_value): 
    y, c = initial_value, C(initial_value, e_R, h_R, b, L_R, h_W, L_W, demand_rate)
    c_new = C(y + 1, e_R, h_R, b, L_R, h_W, L_W, demand_rate)
    while c_new < c:
        c = c_new
        y = y + 1
        c_new = C(y + 1, e_R, h_R, b, L_R, h_W, L_W, demand_rate)
    return y

retailer = {"holding_cost": 1.5, "penalty_cost": 10, "lead_time": 5, "demand_rate": 10}
warehouse = {"holding_cost": 1, "lead_time": 5}

h_W, h_R = warehouse["holding_cost"], retailer["holding_cost"]
e_W = h_W
e_R = h_R - e_W
b, demand_rate = retailer["penalty_cost"], retailer["demand_rate"]
L_R, L_W = retailer["lead_time"], warehouse["lead_time"]

#plt.plot(range(65,80), [C_R(y, e_R, h_R, b, L_R, demand_rate) for y in range(65,80)], label="C_R")
#plt.legend()
#plt.savefig('/Users/gwren/Downloads/6_C_R.eps', format='eps')
#plt.show()

#plt.plot(range(120,150), [C(y, e_R, h_R, b, L_R, h_W, L_W, demand_rate) for y in range(120,150)], label="C")
#plt.legend()
#plt.savefig('/Users/gwren/Downloads/7_C.eps', format='eps')
#plt.show()

initial_value = 100
ye_R = compute_y_R(h_W, h_R, b, L_R, demand_rate)
ye_W = compute_y_W(e_R, h_R, b, L_R, h_W, L_W, demand_rate, initial_value)
print("y^e_R="+str(ye_R))
print("y^e_W="+str(ye_W))
print("y_W="+str(ye_W-ye_R))
print("C(y^e_W)="+str(C(ye_W, e_R, h_R, b, L_R, h_W, L_W, demand_rate)))

