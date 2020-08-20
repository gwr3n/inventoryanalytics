import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt     

from scipy.stats import t

import statsmodels.api as sm 
import pylab as py 

def sample_gaussian_process():
    np.random.seed(1234)
    mu, sigma = 20, 5 # mean and standard deviation
    return np.random.normal(mu, sigma, 200)

def generate_dataframe():
    data = {
        'Value' : sample_gaussian_process()
    }
    return pd.DataFrame(data, columns = ['Value'], index=range(200))

def moving_average(series, n):
    """Calculate rolling average of last n realisations
    """
    return df.rolling(window=n).mean()

def plot_moving_average(series, window, plot_intervals=False, confidence=0.95, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    scale = t.ppf(confidence, len(series) - window - 1, loc=0, scale=1)

    rolling_mean = moving_average(series, window)

    plt.figure(figsize=(7,5))
    plt.title("Moving average\n window size = {}".format(window))
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
    plt.show()

def moving_average_residuals(series, window):
    rolling_mean = moving_average(series, window)
    deviation = df[window:] - rolling_mean[window:]
    return (df[window:]['Value']-rolling_mean[window:]['Value'])/math.sqrt(sum(deviation['Value']**2)/(len(deviation)-1))

def moving_average_residuals_plot(series, window):
    f = plt.figure(1)
    plt.title("Moving average residuals\n window size = {}".format(window))
    plt.plot(moving_average_residuals(series, window), "g", label="Residuals")
    plt.grid(True)
    f.show()

def moving_average_residuals_histogram(series, window):
    f = plt.figure(2)
    plt.title("Moving average residuals\n window size = {}".format(window))
    num_bins = 30
    plt.hist(moving_average_residuals(series, window), num_bins, facecolor='blue', alpha=0.5)
    f.show()

df = generate_dataframe()
window = 32
moving_average_residuals_plot(df, window)
moving_average_residuals_histogram(df, window)
sm.qqplot(moving_average_residuals(df, window), line ='45') 
py.show()
input()
