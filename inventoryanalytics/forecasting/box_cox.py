'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import statsmodels.api as sm, pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import inv_boxcox

def plot_air_passenger_data():
    airpass = sm.datasets.get_rdataset("AirPassengers", "datasets")
    plt.plot(pd.Series(airpass.data["value"]))
    plt.title("Monthly Airline Passenger Numbers 1949-1960, in thousands")
    plt.xlabel('Month')
    plt.savefig('/Users/gwren/Downloads/29_airline.eps', format='eps')
    plt.show()

def plot_air_passenger_data_box_cox():
    airpass = sm.datasets.get_rdataset("AirPassengers", "datasets")
    series, l = stats.boxcox(airpass.data["value"])
    print(l)
    #series = inv_boxcox(series, l) # to obtain the original time series
    plt.title("Monthly Airline Passenger Numbers 1949-1960")
    plt.plot(series)
    plt.xlabel('Month')
    plt.savefig('/Users/gwren/Downloads/30_airline_bc.eps', format='eps')
    plt.show()

#plot_air_passenger_data()
plot_air_passenger_data_box_cox()