import statsmodels.api as sm, pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import inv_boxcox

airpass = sm.datasets.get_rdataset("AirPassengers", "datasets")
#plt.plot(pd.Series(airpass.data["value"]))
series, l = stats.boxcox(airpass.data["value"])
print(l)
series = inv_boxcox(series, l)
plt.title("Monthly Airline Passenger Numbers 1949-1960, in thousands")
plt.plot(series)
plt.show()
