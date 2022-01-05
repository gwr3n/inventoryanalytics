'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

# Importing everything from above

#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
#from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

#def mean_absolute_percentage_error(y_true, y_pred): 
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import math, statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py                  

# plot_moving_average
from sklearn.metrics import mean_absolute_error
from scipy.stats import t

def sample_gaussian_process(mu, sigma, realisations):
    np.random.seed(1234)
    return np.random.normal(mu, sigma, realisations)

def moving_average(series, w, t):
    """Forecasts elements t+1, t+2, ... of series
    """
    forecasts = np.empty(t+1)
    forecasts.fill(np.nan)
    for k in range(1,len(series)-t):
        forecasts = np.append(forecasts, series[t+1-w:t+1].mean())
    return forecasts

def moving_average_rolling(series, w):
    """Calculate rolling average of last n realisations
    """
    return series.rolling(window=w).mean()

def plot(realisations, forecasts, window):
    f = plt.figure(1)
    plt.title("Moving Average forecasts\n window size = {}".format(window))
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Moving Average forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, label="Actual values ($x_t$)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig('/Users/gwren/Downloads/3_sma.svg', format='svg')
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
    plt.savefig('/Users/gwren/Downloads/4_sma_residuals.eps', format='eps')
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.savefig('/Users/gwren/Downloads/5_sma_residuals_histogram.svg', format='svg')
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.acorr(residuals, maxlags=window)
    plt.savefig('/Users/gwren/Downloads/6_sma_residuals_acor.eps', format='eps')
    f.show()

N, t, window = 200, 160, 32
realisations = pd.Series(sample_gaussian_process(20, 5, N), range(N))
forecasts = moving_average(realisations, window, t)
plot(realisations, forecasts, window) 
forecasts = moving_average_rolling(realisations, window)
residuals = residuals(realisations[window:], forecasts[window:])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[window:], forecasts[window:])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
plt.savefig('/Users/gwren/Downloads/7_sma_residuals_QQ.eps', format='eps')
py.show() 

    
  

