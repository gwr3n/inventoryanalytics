import numpy as np, pandas as pd
import matplotlib.pyplot as plt, pylab as py

def plot_series(series, title, xlabel):
    f = plt.figure(1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.plot(series)
    plt.grid(True)
    f.show()

def sample_gaussian_process(mu, sigma, realisations):
    np.random.seed(1234)
    return np.random.normal(mu, sigma, realisations)

# data = [26664.4, 26828.5, 27201.5, 27387.0, 27433.5, 27791.4, 27686.9, 27976.8, 27896.7, 27931.0 ]
# index= pd.date_range(start='2020-08-03', end='2020-08-14', freq='B')
# # index = [pd.Timestamp('2020-08-03'), pd.Timestamp('2020-08-04'), pd.Timestamp('2020-08-05'),
# #          pd.Timestamp('2020-08-06'), pd.Timestamp('2020-08-07'), pd.Timestamp('2020-08-10'),
# #          pd.Timestamp('2020-08-11'), pd.Timestamp('2020-08-12'), pd.Timestamp('2020-08-13'),
# #          pd.Timestamp('2020-08-14')] # alternative to date_range
# plot_series(pd.Series(data, index), "Dow Jones Industrial average", "Day")

data = sample_gaussian_process(0, 1, 30)
plot_series(pd.Series(data, range(30)), "Standard Gaussian noise", "realisation #")
py.show()