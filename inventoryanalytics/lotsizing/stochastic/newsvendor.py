import numpy as np, scipy.integrate as integrate, matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

class Newsvendor:
    def __init__(self, instance):
        self.mean, self.std, self.o, self.u = instance["mean"], instance["std"], instance["o"], instance["u"]

    def crit_frac_solution(self): # critical fractile
        return norm.ppf(self.u/(self.o+self.u), loc=self.mean, scale=self.std)

    def cfolf(self,Q): # complementary first order loss function
        return integrate.quad(lambda x: norm.cdf(x, loc=self.mean, scale=self.std), 0, Q)[0]
    
    def folf(self,Q): # first order loss function
        return self.cfolf(Q)-self.u*(Q - self.mean)

    def C(self,Q): # C(Q)
        return (self.o+self.u)*self.cfolf(Q)-self.u*(Q - self.mean)

    def optC(self): # min C(Q)
        return minimize(self.C, 0, method='Nelder-Mead')

instance = {"o" : 1, "u": 5, "mean" : 10, "std" : 2}
nb = Newsvendor(instance)
print("Q^*=" + str(nb.crit_frac_solution()))
print("C(Q^*)=" + str(nb.C(nb.crit_frac_solution())))
print(nb.optC())

plt.plot([nb.C(Q) for Q in np.arange(7,20,0.1)], label="C(Q)")
plt.plot([nb.folf(Q) for Q in np.arange(7,11,0.1)], label="uE[max(d-Q,0)]")
plt.plot([nb.cfolf(Q) for Q in np.arange(7,20,0.1)], label="oE[max(Q-d,0)]")
plt.legend()
plt.show()