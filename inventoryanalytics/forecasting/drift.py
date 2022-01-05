'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py  

def sample_random_walk(X0, c, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = c + Xt + e
        yield Xt

def drift(series, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(t+1)
    forecasts.fill(np.nan)
    x1 = series[0]
    xt = series[t]
    for k in range(1,len(series)-t):
        xtk = xt+k*(xt-x1)/t
        forecasts = np.append(forecasts, xtk)
    return forecasts

def drift_rolling(series):
    forecasts = np.empty(2)
    forecasts.fill(np.nan)
    for k in range(2,len(series)):
        xk = drift(series[:k+1], k-1)[-1]
        forecasts = np.append(forecasts, xk)
    return forecasts

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Drift method")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "r", label="Drift forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, "b", label="Actual values ($x_t$)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('/Users/gwren/Downloads/13_drift_gaussian.svg', format='svg')
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
    plt.savefig('/Users/gwren/Downloads/14_drift_residuals.eps', format='eps')
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.savefig('/Users/gwren/Downloads/15_drift_residuals_histogram.svg', format='svg')
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.acorr(residuals, maxlags=window)
    plt.savefig('/Users/gwren/Downloads/16_drift_residuals_acor.eps', format='eps')
    f.show()

N, t, window = 200, 160, 2
realisations = pd.Series(list(sample_random_walk(0, 0.1, N)), range(N))
forecasts = drift(realisations, t)
plot(realisations, forecasts) 
forecasts = pd.Series(list(drift_rolling(realisations)), range(N))
residuals = residuals(realisations[window:], forecasts[window:])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[window:], forecasts[window:])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
plt.savefig('/Users/gwren/Downloads/17_drift_residuals_QQ.eps', format='eps')
py.show() 