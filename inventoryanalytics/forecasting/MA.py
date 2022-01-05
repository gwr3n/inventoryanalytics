'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np, pandas as pd, statistics
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm
from statsmodels.tsa.stattools import arma_order_select_ic

def sample_MA_process(mu, theta, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations + len(theta))
    theta = np.r_[1, theta][::-1]
    for r in range(1, min(len(theta),realisations+1)):
        yield mu + sum(np.multiply(theta[-r:], errors[:r]))
    for r in range(realisations-len(theta)+1):
        yield mu + sum(np.multiply(theta, errors[r:r+len(theta)]))

# ARMA(0,1)
def sample_MA_process_ARMA(mu, theta, realisations):
    np.random.seed(1234)
    dist = lambda size: np.random.normal(0, 1, size)
    arparams = np.array([])
    maparams = np.array(theta)
    # include zero-th lag
    arparams = np.r_[1, arparams]
    maparams = np.r_[1, maparams]
    arma_t = ArmaProcess(arparams, maparams)
    return arma_t.generate_sample(nsample=realisations, distrvs=dist)  

def simulate_MA():
    mu, theta, N = 0, [0.8,0.2], 200
    realisations = pd.Series(list(sample_MA_process(mu, theta, N)), range(N))
    print(realisations)
    f = plt.figure(1)
    plt.plot(realisations)
    f.show()

    f = plt.figure(2)
    realisations = pd.Series(list(sample_MA_process_ARMA(mu, theta, N)), range(N))
    print(realisations)
    plt.plot(realisations)
    f.show()
    plt.show()

def fit_MA_q():
    mu, theta, N, t, max_order = 0, [0.8,0.2], 200, 180, 10
    realisations = pd.Series(list(sample_MA_process_ARMA(mu, theta, N)), range(N))

    order = arma_order_select_ic(realisations[0:t], ic='aic', max_ar = 5, max_ma = 5)
    print(order.aic_min_order)

    mod = sm.tsa.ARMA(realisations[0:t], order=(0, 2))
    res = mod.fit()
    print(res.summary())
    print("Std residuals: "+str(statistics.stdev(res.resid)))
    
    sm.graphics.tsa.plot_acf(realisations.values.squeeze(), lags=max_order)
    plt.xlabel('lag (in periods)', fontsize=13)
    plt.ylabel('ACF', fontsize=13)
    plt.title('')
    plt.savefig('/Users/gwren/Downloads/45_ma_acf.svg', format='svg')

    f = plt.figure(1)
    res.plot_predict(start=t, end=N, plot_insample=False)
    plt.xlabel('Period ($t$)')
    plt.plot(realisations[0:N], label="realisations")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()
    plt.savefig('/Users/gwren/Downloads/46_ma_2_fit_forecasts.svg', format='svg')
    plt.show()

fit_MA_q()