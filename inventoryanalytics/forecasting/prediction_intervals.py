'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import math
import numpy as np, pandas as pd
from scipy.stats import t
from scipy.stats import norm
import statistics as s
import matplotlib.pyplot as plt

def confidence_intervals():
    np.random.seed(1234)
    replications = 100
    n = 30
    x = range(replications)
    y = np.random.normal(0, 1, size=(replications, n)) # realisations
    alpha = 0.95 # confidence level
    z = t.ppf(1-(1-alpha)/2, n-1) # inverse t distribution
    y_mean = [s.mean(y[r]) for r in range(replications)]
    e = [z*s.stdev(y[r])/math.sqrt(n) for r in range(replications)]
    ec = ['red' if (y_mean[r]+z*s.stdev(y[r])/math.sqrt(n) < 0 or 
                    y_mean[r]-z*s.stdev(y[r])/math.sqrt(n) > 0) 
                else 'black' for r in range(replications)]            
    plt.errorbar(x, y_mean, yerr=e, ecolor=ec, fmt='none')
    plt.grid(True)
    plt.xlabel('Replication', fontsize=13)
    plt.ylabel('$\mathcal{I}(\\alpha)$', rotation=0, fontsize=13)
    plt.savefig('/Users/gwren/Downloads/25_confidence_intervals.eps', format='eps')
    plt.show()

def prediction_intervals_known_parameters():
    np.random.seed(4321)
    replications = 100
    x = range(replications)
    mu, sigma = 10, 2
    y = np.random.normal(mu, sigma, replications) # realisations
    alpha = 0.95 # confidence level
    z = norm.ppf(1-(1-alpha)/2) # inverse standard normal distribution
    plt.plot(x,[mu-z*sigma for k in x], color='blue', linestyle='dashed')
    plt.plot(x,[mu+z*sigma for k in x], color='blue', linestyle='dashed')
    ec = ['red' if y[r]>mu+z*sigma or y[r]<mu-z*sigma 
                else 'blue' for r in range(replications)]
    plt.scatter(x, y, color=ec)
    plt.grid(True)
    plt.xlabel('$t$', fontsize=15)
    plt.ylabel('$X_t$ ', rotation=0, fontsize=15, loc='top')
    plt.savefig('/Users/gwren/Downloads/26_normal_prediction_intervals.eps', format='eps')
    plt.show()

def prediction_intervals_unknown_parameters():
    np.random.seed(4321)
    replications = 100
    mu, sigma = 10, 2
    x = range(replications)
    y = np.random.normal(mu, sigma, replications) # realisations
    alpha = 0.95 # confidence level
    z = lambda n: t.ppf(1-(1-alpha)/2, n-1) # inverse t distribution
    y_mean = [s.mean(y[0:r+1]) for r in range(replications)]
    e = [z(r-1)*s.stdev(y[0:r+1])*math.sqrt(1+1/r) 
        if r > 2 else 30*sigma for r in range(replications)]
    plt.errorbar(x[:-1], y_mean[:-1], yerr=e[:-1], fmt='none')
    ec = ['red' if y[1:][r]>y_mean[:-1][r]+e[:-1][r] or 
                y[1:][r]<y_mean[:-1][r]-e[:-1][r]
                else 'blue' for r in range(replications-1)]
    plt.scatter(x[:-1], y[1:], color=ec)
    plt.grid(True)
    plt.xlabel('$t$')
    plt.ylabel('$X_t$', rotation=0)
    plt.savefig('/Users/gwren/Downloads/27_prediction_intervals_unknown_mu_sigma.eps', format='eps')
    plt.show()

def sample_random_walk(X0, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = Xt + e
        yield Xt

def naive(series, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[:t+1] = np.nan
    forecasts[t+1:] = series[t]
    return forecasts

def naive_rolling(series):
    return series.shift(periods=1)

def residuals(realisations, forecasts):
    return realisations - forecasts

def plot(realisations, forecasts, stdev, alpha):
    f = plt.figure(1)
    plt.title("Naïve method")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "r", label="Naïve forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, "b", label="Actual values ($x_t$)")
    z = t.ppf(1-(1-alpha)/2, len(realisations)-1) # inverse t distribution
    plt.fill_between(range(first, last+1), 
                     [forecasts[first+k]-z*stdev*math.sqrt(k) for k in range(last-first+1)], 
                     [forecasts[first+k]+z*stdev*math.sqrt(k) for k in range(last-first+1)],
                     color='r', alpha=0.1)
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def prediction_intervals_naive():
    N, t, window, alpha = 200, 160, 1, 0.95
    realisations = pd.Series(list(sample_random_walk(0, N)), range(N))
    forecasts = naive(realisations, t)
    forecasts_roll = naive_rolling(realisations)
    res = residuals(realisations[window:], forecasts_roll[window:])
    plot(realisations, forecasts, s.stdev(res), alpha) 
    print("E[e_t] = "+str(s.mean(res)))
    print("Stdev[e_t] = "+str(s.stdev(res)))
    plt.savefig('/Users/gwren/Downloads/28_prediction_intervals_naive.svg', format='svg')
    plt.show()

#confidence_intervals()
prediction_intervals_known_parameters()
#prediction_intervals_unknown_parameters()
#prediction_intervals_naive()
