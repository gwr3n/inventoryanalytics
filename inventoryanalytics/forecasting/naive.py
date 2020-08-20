import statistics
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt  

import statsmodels.api as sm 
import pylab as py 

def sample_random_walk(X0, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = Xt + e
        yield Xt

def naive(series):
    return series.shift(periods=1)

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Naïve method")
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.plot(forecasts, "r", label="Naïve forecasts")
    plt.plot(realisations[1:], "b", label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def residuals(realisations, forecasts):
    return realisations - forecasts

def standardised_residuals(realisations, forecasts):
    residuals = realisations - forecasts
    return (residuals) / statistics.stdev(residuals)

def residuals_plot(residuals):
    f = plt.figure(2)
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.plot(residuals, "g", label="Residuals")
    plt.grid(True)
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Period')
    plt.ylabel('Value')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.acorr(residuals, maxlags=window)
    f.show()

window = 1
realisations = pd.DataFrame({'Value' : list(sample_random_walk(0, 200))}, columns = ['Value'], index=range(200))
forecasts = naive(realisations)
plot(realisations, forecasts) 
residuals = residuals(realisations[window:]['Value'], forecasts[window:]['Value'])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[window:]['Value'], forecasts[window:]['Value'])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
py.show() 
