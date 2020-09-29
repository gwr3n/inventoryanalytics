import statsmodels.api as sm, pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

N, t = 140, 136
airpass = sm.datasets.get_rdataset("AirPassengers", "datasets")
ts = pd.Series(airpass.data["value"])
ts = ts.astype(float)
model = ARIMA(ts[0:t], order=(0,1,0))
fitted = model.fit()
fitted.plot_predict(start=ts.index[3], end=ts.index[-1], alpha=0.1)
plt.show()