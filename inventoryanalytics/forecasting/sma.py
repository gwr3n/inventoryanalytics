# Importing everything from above

#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
#from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

#def mean_absolute_percentage_error(y_true, y_pred): 
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import math
import statistics
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt                  

# plot_moving_average
from sklearn.metrics import mean_absolute_error
from scipy.stats import t

# Q-Q plot
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

def moving_average_residuals(series, window):
    rolling_mean = moving_average(series, window)
    deviation = df[window:] - rolling_mean[window:]
    return deviation['Value']

def moving_average_standardised_residuals(series, window):
    rolling_mean = moving_average(series, window)
    deviation = df[window:] - rolling_mean[window:]
    return deviation['Value'] / statistics.stdev(deviation['Value'])

def moving_average_residuals_plot(series, window):
    f = plt.figure(2)
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.plot(moving_average_standardised_residuals(series, window), "g", label="Residuals")
    plt.grid(True)
    f.show()

def moving_average_residuals_histogram(series, window):
    f = plt.figure(3)
    plt.xlabel('Period')
    plt.ylabel('Value')
    num_bins = 30
    plt.hist(moving_average_standardised_residuals(series, window), num_bins, facecolor='blue', alpha=0.5)
    f.show()

def moving_average_residuals_autocorrelation(series, window):
    f = plt.figure(4)
    plt.acorr(moving_average_standardised_residuals(series, window), maxlags=window)
    f.show()

if __name__ == '__main__':
    df, window = generate_dataframe(), 32
    plot_moving_average(df, window, plot_intervals=False,plot_anomalies=False) 
    residuals = moving_average_residuals(df, window)
    print(statistics.mean(residuals))
    print(statistics.stdev(residuals))
    moving_average_residuals_plot(df, window)
    moving_average_residuals_histogram(df, window)
    moving_average_residuals_autocorrelation(df, window)
    sm.qqplot(moving_average_standardised_residuals(df, window), line ='45') 
    py.show() 

    
  

