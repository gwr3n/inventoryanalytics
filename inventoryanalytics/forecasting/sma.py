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

# def plot_moving_average(series, window, plot_intervals=False, confidence=0.95, plot_anomalies=False):
#     """
#         series - dataframe with timeseries
#         window - rolling window size 
#         plot_intervals - show confidence intervals
#         plot_anomalies - show anomalies 

#     """
#     f = plt.figure(1)
#     scale = t.ppf(confidence, len(series) - window - 1, loc=0, scale=1)

#     rolling_mean = moving_average_rolling(series, window)

#     plt.title("Moving average\n window size = {}".format(window))
#     plt.xlabel('Period')
#     plt.ylabel('xt')
#     plt.plot(rolling_mean, "g", label="Rolling mean trend")

#     # Plot confidence intervals for smoothed values
#     if plot_intervals:
#         mae = mean_absolute_error(series[window:], rolling_mean[window:])
#         deviation = np.std(series[window:] - rolling_mean[window:])
#         lower_bond = rolling_mean - (mae + scale * deviation * math.sqrt(1+1/(len(series) - window)))
#         upper_bond = rolling_mean + (mae + scale * deviation * math.sqrt(1+1/(len(series) - window)))
#         plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
#         plt.plot(lower_bond, "r--")
        
#         # Having the intervals, find abnormal values
#         if plot_anomalies:
#             anomalies = pd.DataFrame(index=series.index, columns=series.columns)
#             anomalies[series<lower_bond] = series[series<lower_bond]
#             anomalies[series>upper_bond] = series[series>upper_bond]
#             plt.plot(anomalies, "ro", markersize=10)
        
#     plt.plot(series[window:], label="Actual values")
#     plt.legend(loc="upper left")
#     plt.grid(True)
#     f.show()

def plot(realisations, forecasts, window):
    f = plt.figure(1)
    plt.title("Moving Average forecasts\n window size = {}".format(window))
    plt.xlabel('Period')
    first, last = next(x for x, val in enumerate(forecasts) if ~np.isnan(val)), len(forecasts)-1
    plt.axvspan(first, last, alpha=0.2, color='blue')
    plt.plot(forecasts, "g", label="Moving Average forecasts")
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
py.show() 

    
  

