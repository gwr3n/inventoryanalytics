'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import math, statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py  

from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def sample_seasonal_random_walk(realisations, m):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = errors[:m]
    for t in range(m,realisations):
        Xt = np.append(Xt, Xt[t-m] + errors[t])
    return Xt

def moving_average(series, w, t):
    """Forecasts elements t+1, t+2, ... of series
    """
    forecasts = np.empty(t+1)
    forecasts.fill(np.nan)
    for k in range(1,len(series)-t):
        forecasts = np.append(forecasts, series[t+1-w:t+1].mean())
    return forecasts

def naive(series, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[:t+1] = np.nan
    forecasts[t+1:] = series[t]
    return forecasts

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

def seasonal_naive(series, m, t):
    """Forecasts periods t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[:t+1] = np.nan
    for k in range(t+1,len(series)):
        forecasts[k] = series[k-m*((k-t-1)//m+1)]
    return forecasts

def plot(realisations, forecasts, test_window):
    f = plt.figure(1)
    plt.xlabel('Period ($t$)')
    plt.ylabel('Realisation ($x_t$)')
    plt.axvspan(*test_window, alpha=0.2, color='blue')
    plt.plot(realisations, "b")
    plt.text(83, -5, 'Test data')
    plt.text(35, -5, 'Training data')
    plt.grid(True)
    f.show()

def plot_methods(realisations, forecasts, test_window):
    f = plt.figure(2)
    plt.xlabel('Period ($t$)')
    plt.plot(realisations, label="Actual values")
    plt.axvspan(*test_window, alpha=0.2, color='blue')
    for key in forecasts:
        plt.plot(forecasts[key], label=key)
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

N, t, window, m, test_window = 100, 80, 5, 5, [81,100]
realisations = pd.Series(list(sample_seasonal_random_walk(N, m)), range(N))
sma_forecasts = moving_average(realisations, window, t)
naive_forecasts = naive(realisations, t)
drift_forecasts = drift(realisations, t)
seasonal_naive_forecasts = seasonal_naive(realisations, m, t)
plot(realisations, seasonal_naive_forecasts, test_window) 
plt.savefig('/Users/gwren/Downloads/23_training_test_data.svg', format='svg')
methods = {
    "Moving Average": sma_forecasts, 
    "Naive": naive_forecasts, 
    "Drift": drift_forecasts, 
    "Seasonal naive": seasonal_naive_forecasts}
plot_methods(realisations, methods, test_window) 
plt.savefig('/Users/gwren/Downloads/24_seasonal_random_walk_forecasting_methods.svg', format='svg')
print("MAE")
for k in methods:
    print(k,end=':\t')
    print(mean_absolute_error(realisations[t+1:],methods[k][t+1:]))
print("\nMSE")
for k in methods:
    print(k,end=':\t')
    print(mean_squared_error(realisations[t+1:],methods[k][t+1:]))
print("\nRMSE")
for k in methods:
    print(k,end=':\t')
    print(math.sqrt(mean_squared_error(realisations[t+1:],methods[k][t+1:])))
print("\nMAPE")
for k in methods:
    print(k,end=':\t')
    print(mean_absolute_percentage_error(realisations[t+1:],methods[k][t+1:]))
py.show() 

