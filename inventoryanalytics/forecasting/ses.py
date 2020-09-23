import statistics, scipy.stats as stats, statsmodels.api as sm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py   

from statsmodels.tsa.api import SimpleExpSmoothing

from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

def sample_gaussian_process(mu, sigma, realisations):
    np.random.seed(1234)
    return np.random.normal(mu, sigma, realisations)

def sample_random_walk(X0, realisations):
    np.random.seed(1234)
    errors = np.random.normal(0, 1, realisations)
    Xt = X0
    for e in errors:
        Xt = Xt + e
        yield Xt

def ses(series, alpha, x0, t):
    """Forecasts elements t+1, t+2, ... of series
    """
    forecasts = np.empty(len(series))
    forecasts[0] =  x0
    for k in range(1,t+2):
        forecasts[k] = alpha*series[k-1] + (1-alpha)*forecasts[k-1]
    for k in range(t+2,len(series)):
        forecasts[k] = forecasts[k-1]
    forecasts[0:t] = np.nan
    return forecasts

def ses_rolling(series, alpha, x0):
    forecasts = np.empty(len(series))
    forecasts[0] =  x0
    for k in range(1,len(series)):
        forecasts[k] = alpha*series[k-1] + (1-alpha)*forecasts[k-1]
    return forecasts

def plot(realisations, forecasts, alpha):
    f = plt.figure(1)
    plt.title("Simple Exponential Smoothing forecasts\n alpha = {}".format(alpha))
    plt.xlabel('Period')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Simple Exponential Smoothing forecasts")
    plt.plot(realisations, label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def residuals(realisations, forecasts):
    return realisations - forecasts

def standardised_residuals(realisations, forecasts):
    residuals = realisations - forecasts
    return (residuals) / statistics.stdev(residuals)

def residuals_plot(residuals):
    f = plt.figure(2)
    plt.xlabel('Period')
    plt.plot(residuals, "g", label="Residuals")
    plt.grid(True)
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Period')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.acorr(residuals, maxlags=window)
    f.show()

def simple_exponential_smoothing():
    N, t, alpha, x0 = 200, 160, 0.5, 20
    realisations = pd.Series(sample_gaussian_process(20, 5, N), range(N))
    forecasts = ses(realisations, alpha, x0, t)
    plot(realisations, forecasts, alpha) 
    forecasts = ses_rolling(realisations, alpha, x0)
    res = residuals(realisations, forecasts)
    print("E[e_t] = "+str(statistics.mean(res)))
    print("Stdev[e_t] = "+str(statistics.stdev(res)))
    standardised_res = standardised_residuals(realisations, forecasts)
    residuals_plot(res)
    residuals_histogram(standardised_res)
    residuals_autocorrelation(res, None)
    sm.qqplot(standardised_res, line ='45') 
    py.show() 

def simple_exponential_smoothing_statsmodels():
    N, t, alpha, x0 = 200, 160, 0.5, 20
    realisations = pd.Series(sample_gaussian_process(20, 5, N), range(N))
    mod = SimpleExpSmoothing(realisations[:t+1]).fit(smoothing_level=alpha, initial_level=x0, optimized=False)
    forecasts = mod.forecast(N-(t+1)).rename(r'$\alpha=0.5$')
    plot(realisations, pd.Series(np.nan, range(t+1)).append(forecasts), alpha)
    py.show()

def plot_ci(realisations, forecasts, forecasts_ci, alpha):
    f = plt.figure(1)
    plt.title("Simple Exponential Smoothing forecasts\n State Space Model")
    plt.xlabel('Period')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Simple Exponential Smoothing forecasts")
    plt.plot(realisations, label="Actual values")
    t = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)) - 1
    forecast_index = np.arange(t+1, t+1 + len(forecasts_ci))
    plt.fill_between(forecast_index, forecasts_ci.iloc[:, 0], forecasts_ci.iloc[:, 1], color='r', alpha=0.1)
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def simple_exponential_smoothing_ci(stochastic_process):
    N, t, alpha = 200, 160, 0.5
    if stochastic_process == "GP":
        x0 = 20
        realisations = pd.Series(sample_gaussian_process(20, 5, N), range(N))
    elif stochastic_process == "RW":
        x0 = 0
        realisations = pd.Series(list(sample_random_walk(0, N)), range(N))
    else:
        quit()
    mod = ExponentialSmoothing(realisations[:t+1], initialization_method='known', initial_level=x0).fit(disp=False)
    print(mod.summary())
    forecasts = mod.get_forecast(N-(t+1))
    forecasts_ci = forecasts.conf_int(alpha=0.05)
    plot_ci(realisations, pd.Series(np.nan, range(t+1)).append(forecasts.predicted_mean), forecasts_ci, alpha)
    py.show()

# simple_exponential_smoothing()
# simple_exponential_smoothing_statsmodels()
simple_exponential_smoothing_ci("RW")
