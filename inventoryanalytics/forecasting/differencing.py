'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import statsmodels.api as sm, pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.statespace.tools import diff

airpass = sm.datasets.get_rdataset("AirPassengers", "datasets")
fig, axs = plt.subplots(3)
axs[0].set_title('Monthly Airline Passenger Numbers 1949-1960, in thousands')
axs[0].plot(pd.Series(airpass.data["value"]))
series, l = stats.boxcox(airpass.data["value"])
axs[1].plot(series)
axs[1].set_title('Box Cox Transformation')
differenced = diff(series, k_diff=12)
axs[2].plot(differenced)
axs[2].set_title('Seasonally differenced (m=12)')
plt.xlabel('Period ($t$)')
fig.tight_layout()
plt.savefig('/Users/gwren/Downloads/42_airline_bc_sd.eps', format='eps')
plt.show()