import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py
from statsmodels.tsa.api import ExponentialSmoothing
#from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

def sample_seasonal_random_walk(realisations, m):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = errors[:m]
    for t in range(m,realisations):
        Xt = np.append(Xt, Xt[t-m] + errors[t])
    return Xt

def plot(realisations, forecasts):
    f = plt.figure(1)
    plt.title("Holt-Winters' forecasts")
    plt.xlabel('Period')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(realisations, label="Actual values")
    plt.plot(forecasts, "g", label="Holt-Winters' forecasts")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def plot_components(fit):
    f = plt.figure(1)
    pd.DataFrame(np.c_[fit.level,fit.slope,fit.season]).rename(
        columns={0:'level',1:'slope',2:'seasonal'}).plot(subplots=True)
    f.show()

# uncomment from statsmodels.tsa.api import ExponentialSmoothing
def holt_winters():
    N, t, m = 100, 80, 4
    realisations = pd.Series(list(sample_seasonal_random_walk(N,m)), range(N))
    mod = ExponentialSmoothing(realisations[:t+1], seasonal_periods=4, trend='add', seasonal='add').fit(optimized=True)
    params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
    results=pd.DataFrame(index=["alpha","beta","gamma","l_0","b_0","SSE"] ,columns=["Holt-Winters'"])
    results["Holt-Winters'"] = [mod.params[p] for p in params] + [mod.sse]
    print(results)
    forecasts = mod.forecast(N-(t+1)).rename(r'$\alpha=0.5$ and $\beta=0.5$')
    plot(realisations, pd.Series(np.nan, range(t+1)).append(forecasts))
    plot_components(mod)
    py.show()

def plot_ci(realisations, forecasts, forecasts_ci):
    f = plt.figure(1)
    plt.title("Holt-Winters' forecasts\n State Space Model")
    plt.xlabel('Period')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(realisations, label="Actual values")
    plt.plot(forecasts, "g", label="Holt-Winters' forecasts")
    t = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)) - 1
    forecast_index = np.arange(t+1, t+1 + len(forecasts_ci))
    plt.fill_between(forecast_index, forecasts_ci.iloc[:, 0], forecasts_ci.iloc[:, 1], color='r', alpha=0.2)
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

# uncomment from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
def holt_winters_ci():
    N, t, m = 100, 80, 4
    realisations = pd.Series(list(sample_seasonal_random_walk(N,m)), range(N))
    mod = ExponentialSmoothing(realisations[:t+1], trend=True, seasonal=m, initialization_method='estimated').fit(disp=False)
    print(mod.summary())
    forecasts = mod.get_forecast(N-(t+1))
    forecasts_ci = forecasts.conf_int(alpha=0.05)
    plot_ci(realisations, pd.Series(np.nan, range(t+1)).append(forecasts.predicted_mean), forecasts_ci)
    py.show()

holt_winters()
#holt_winters_ci()