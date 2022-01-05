'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py  

from statsmodels.tsa.arima_process import ArmaProcess

def sample_random_walk(X0, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = Xt + e
        yield Xt

# ARMA(1,0) random walk
def sample_random_walk_arma(X0, realisations):
    np.random.seed(1234)
    # ARMA(1,1)
    arparams = np.array([1])
    maparams = np.array([0])
    # include zero-th lag
    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, maparams]
    arma_t = ArmaProcess(arparams, maparams)
    return arma_t.generate_sample(nsample=realisations)       

def naive(series, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[:t+1] = np.nan
    forecasts[t+1:] = series[t]
    return forecasts

def naive_rolling(series):
    return series.shift(periods=1)

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Naïve method")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "r", label="Naïve forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, "b", label="Actual values ($x_t$)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('/Users/gwren/Downloads/8_naive.svg', format='svg')
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
    plt.savefig('/Users/gwren/Downloads/9_naive_residuals.eps', format='eps')
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.savefig('/Users/gwren/Downloads/10_naive_residuals_histogram.svg', format='svg')
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.acorr(residuals, maxlags=window)
    plt.savefig('/Users/gwren/Downloads/11_naive_residuals_acor.eps', format='eps')
    f.show()

N, t, window = 200, 160, 1
realisations = pd.Series(list(sample_random_walk(0, N)), range(N))
forecasts = naive(realisations, t)
plot(realisations, forecasts) 
forecasts = naive_rolling(realisations)
residuals = residuals(realisations[window:], forecasts[window:])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[window:], forecasts[window:])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
plt.savefig('/Users/gwren/Downloads/12_naive_residuals_QQ.eps', format='eps')
py.show() 
