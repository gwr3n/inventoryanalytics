from itertools import accumulate
from scipy.stats import poisson
import scipy.integrate as integrate
from scipy.optimize import minimize

class MultiPeriodNewsvendor:
    def __init__(self, instance):
        self.mean, self.o, self.u = instance["mean"], instance["o"], instance["u"]    

    def cfolf(self, Q, d): # complementary first order loss function
        return integrate.quad(lambda x: poisson.cdf(x, d), 0, Q)[0]

    def folf(self,Q): # first order loss function
        return self.cfolf(Q)-self.u*(Q - self.mean)

    def C(self, Q): # C(Q)
        return sum([(self.o+self.u)*self.cfolf(Q, d)-self.u*(Q - d) for d in accumulate(self.mean)])

    def optC(self): # min C(Q)
        return minimize(self.C, 0, method='Nelder-Mead')

    def verify_fractile_solution(self, Q):
        T = len(self.mean)
        critical_fractile = T*self.u/(self.u+self.o)
        return sum([poisson.cdf(Q, d) for d in accumulate(self.mean)]) - critical_fractile < 0.1

instance = {"o" : 1, "u": 5, "mean" : [10,10,10]}
nb = MultiPeriodNewsvendor(instance)
res = nb.optC()
print(res)
print("Optimal cost: "+str(res.fun))
print("Verify critical fractile solution: "+str(nb.verify_fractile_solution(res.x[0])))
