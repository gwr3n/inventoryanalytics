# Importing everything from above

#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
#from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

#def mean_absolute_percentage_error(y_true, y_pred): 
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import math
import statistics
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt                  

# plot_moving_average
from sklearn.metrics import mean_absolute_error
from scipy.stats import t

# Q-Q plot
import statsmodels.api as sm 
import pylab as py 

def sample_gaussian_process(mu, sigma, realisations):
    np.random.seed(1234)
    return np.random.normal(mu, sigma, realisations)

def moving_average(series, n):
    """Calculate rolling average of last n realisations
    """
    return series.rolling(window=n).mean()

def plot_moving_average(series, window, plot_intervals=False, confidence=0.95, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    f = plt.figure(1)
    scale = t.ppf(confidence, len(series) - window - 1, loc=0, scale=1)

    rolling_mean = moving_average(series, window)

    plt.title("Moving average\n window size = {}".format(window))
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation * math.sqrt(1+1/(len(series) - window)))
        upper_bond = rolling_mean + (mae + scale * deviation * math.sqrt(1+1/(len(series) - window)))
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    f.show()

def plot(realisations, forecasts, window):
    f = plt.figure(1)
    plt.title("Moving average\n window size = {}".format(window))
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.plot(forecasts, "g", label="Rolling mean trend")
    plt.plot(realisations[window:], label="Actual values")
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
    plt.ylabel('Value')
    plt.plot(residuals, "g", label="Residuals")
    plt.grid(True)
    f.show()

def residuals_histogram(residuals):
    f = plt.figure(3)
    plt.xlabel('Period')
    plt.ylabel('Value')
    num_bins = 30
    plt.hist(residuals, num_bins, facecolor='blue', alpha=0.5, density=True)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    f.show()

def residuals_autocorrelation(residuals, window):
    f = plt.figure(4)
    plt.acorr(residuals, maxlags=window)
    f.show()

window = 32
realisations = pd.DataFrame({'Value' : sample_gaussian_process(20, 5, 200)}, columns = ['Value'], index=range(200))
forecasts = moving_average(realisations, window)
plot(realisations, forecasts, window) 
residuals = residuals(realisations[window:]['Value'], forecasts[window:]['Value'])
print("E[e_t] = "+str(statistics.mean(residuals)))
print("Stdev[e_t] = "+str(statistics.stdev(residuals)))
standardised_residuals = standardised_residuals(realisations[window:]['Value'], forecasts[window:]['Value'])
residuals_plot(residuals)
residuals_histogram(standardised_residuals)
residuals_autocorrelation(residuals, None)
sm.qqplot(standardised_residuals, line ='45') 
py.show() 

    
  

