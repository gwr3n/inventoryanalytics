'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py
from statsmodels.tsa.api import Holt
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

def sample_random_walk(X0, c, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = c + Xt + e
        yield Xt

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Holt's forecasts")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Holt's forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, label="Actual values ($x_t$)")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def plot_components(fit):
    f = plt.figure(1)
    pd.DataFrame(np.c_[fit.level,fit.trend]).rename(
        columns={0:'level',1:'trend'}).plot(subplots=True)
    plt.xlabel('Period ($t$)')    
    f.show()

def holt():
    N, t = 200, 160
    realisations = pd.Series(list(sample_random_walk(0, 0.1, N)), range(N))
    mod = Holt(realisations[:t+1]).fit(optimized=True)
    params = ['smoothing_level', 'smoothing_trend', 'initial_level', 'initial_trend']
    results=pd.DataFrame(index=["alpha","beta","l_0","b_0","SSE"] ,columns=["Holt's"])
    results["Holt's"] = [mod.params[p] for p in params] + [mod.sse]
    print(results)
    forecasts = mod.forecast(N-(t+1)).rename(r'$\alpha=0.5$ and $\beta=0.5$')
    plot(realisations, pd.Series(np.nan, range(t+1)).append(forecasts))
    plot_components(mod)
    plt.savefig('/Users/gwren/Downloads/38_holt_level_slope.svg', format='svg')
    py.show()

def plot_ci(realisations, forecasts, forecasts_ci):
    f = plt.figure(1)
    plt.title("Holt's forecasts\n State Space Model")
    plt.xlabel('Period ($t$)')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Holt's forecasts ($\widehat{X}_t$)")
    plt.plot(realisations, label="Actual values ($x_t$)")
    t = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)) - 1
    forecast_index = np.arange(t+1, t+1 + len(forecasts_ci))
    plt.fill_between(forecast_index, forecasts_ci.iloc[:, 0], forecasts_ci.iloc[:, 1], color='r', alpha=0.1)
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def holt_ci():
    N, t = 200, 160
    realisations = pd.Series(list(sample_random_walk(0, 0.1, N)), range(N))
    mod = ExponentialSmoothing(realisations[:t+1], trend=True, initialization_method='estimated').fit(disp=False)
    print(mod.summary())
    forecasts = mod.get_forecast(N-(t+1))
    forecasts_ci = forecasts.conf_int(alpha=0.05)
    plot_ci(realisations, pd.Series(np.nan, range(t+1)).append(forecasts.predicted_mean), forecasts_ci)
    plt.savefig('/Users/gwren/Downloads/39_holt_forecasts_pi.svg', format='svg')
    py.show()

holt()
#holt_ci()