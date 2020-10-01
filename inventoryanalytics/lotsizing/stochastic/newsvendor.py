from scipy.stats import norm
import scipy.integrate as integrate
from scipy.optimize import minimize

class Newsvendor:
    def __init__(self, instance):
        self.mean, self.std, self.o, self.u = instance["mean"], instance["std"], instance["o"], instance["u"]

    def crit_frac_solution(self): # critical fractile
        return norm.ppf(self.u/(self.o+self.u), loc=self.mean, scale=self.std)

    def cfolf(self,Q): # complementary first order loss function
        return integrate.quad(lambda x: norm.cdf(x, loc=self.mean, scale=self.std), 0, Q)[0]

    def C(self,Q): # C(Q)
        return (self.o+self.u)*self.cfolf(Q)-self.u*(Q - self.mean)

    def optC(self): # min C(Q)
        return minimize(self.C, 0, method='Nelder-Mead')

instance = {"o" : 1, "u": 5, "mean" : 10, "std" : 2}
nb = Newsvendor(instance)
print("Q^*=" + str(nb.crit_frac_solution()))
print("C(Q^*)=" + str(nb.C(nb.crit_frac_solution())))
print(nb.optC())