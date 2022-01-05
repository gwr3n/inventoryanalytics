'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np, pandas as pd, statistics
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import matplotlib.pyplot as plt

def sample_random_walk(X0, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = Xt + e
        yield Xt

def fit_AR_1():
    N, t, p = 200, 180, 1
    realisations = pd.Series(list(sample_random_walk(0, N)), range(N))
    mod = AutoReg(realisations[0:t], p)
    res = mod.fit()
    print(res.summary())
    print("Std residuals: "+str(statistics.stdev(res.resid)))

    f = plt.figure(1)
    fig = plt.figure(figsize=(16,9))
    res.plot_diagnostics(fig=fig, lags=30)
    f.show()

    f = plt.figure(1)
    res.plot_predict(start=t, end=N)
    plt.plot(realisations[0:N], label="realisations")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def fit_AR_p():
    N, t, p, max_order = 200, 180, 1, 10
    realisations = pd.Series(list(sample_random_walk(0, N)), range(N))
    sel = ar_select_order(realisations[0:t], max_order)
    res = sel.model.fit()
    print(res.summary())
    print("Std residuals: "+str(statistics.stdev(res.resid)))

    f = plt.figure(1)
    res.plot_diagnostics(fig=f, lags=30)
    plt.tight_layout()
    plt.savefig('/Users/gwren/Downloads/43_ar_1_fit_diagnostics.svg', format='svg')
    f.show()

    f = plt.figure(1)
    res.plot_predict(start=t, end=N)
    plt.plot(realisations[0:N], label="realisations")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.xlabel('Period ($t$)')
    plt.savefig('/Users/gwren/Downloads/44_ar_1_fit_forecasts.svg', format='svg')
    f.show()

fit_AR_p()
plt.show()