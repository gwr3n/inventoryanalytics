'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py  

def sample_seasonal_random_walk(realisations, m):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = errors[:m]
    for t in range(m,realisations):
        Xt = np.append(Xt, Xt[t-m] + errors[t])
    return Xt

def seasonal_naive(series, m, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[:t+1] = np.nan
    for k in range(t+1,len(series)):
        forecasts[k] = series[k-m*((k-t-1)//m+1)]
    return forecasts

def seasonal_naive_rolling(series, m):
    forecasts = np.empty(m)
    forecasts.fill(np.nan)
    for k in range(m,len(series)):
        xk = seasonal_naive(series[:k+1], m, k-1)[-1]
        forecasts = np.append(forecasts, xk)
    return forecasts

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Seasonal naive method")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "r", label="Seasonal naive forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, "b", label="Actual values ($x_t$)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('/Users/gwren/Downloads/18_seasonal_naive.svg', format='svg')
    f.show()

def residuals(realisations, forecasts):
    return realisations - forecasts

def standardised_residuals(realisations, forecasts):
    residuals = realisations - forecasts
    return (residuals) / statistics.stdev(residuals)

def residuals_plot(residuals):
    f = plt.figure(2)
    plt.xlabel('Period ($t$)')
    plt.ylabel('Residual')
    plt.plot(residuals, "g", label="Residuals")
    plt.grid(True)
    plt.savefig('/Users/gwren/Downloads/19_seasonal_naive_residuals.eps', format='eps')
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.savefig('/Users/gwren/Downloads/20_seasonal_naive_residuals_histogram.svg', format='svg')
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.acorr(residuals, maxlags=window)
    plt.savefig('/Users/gwren/Downloads/21_seasonal_naive_residuals_acor.eps', format='eps')
    f.show()

N, t, m = 100, 80, 5
realisations = pd.Series(list(sample_seasonal_random_walk(N, m)), range(N))
forecasts = seasonal_naive(realisations, m, t)
plot(realisations, forecasts) 
forecasts = pd.Series(list(seasonal_naive_rolling(realisations, m)), range(N))
residuals = residuals(realisations[m:], forecasts[m:])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[m:], forecasts[m:])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
plt.savefig('/Users/gwren/Downloads/22_seasonal_naive_residuals_QQ.eps', format='eps')
py.show() 