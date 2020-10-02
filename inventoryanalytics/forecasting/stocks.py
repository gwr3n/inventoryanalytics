import yfinance as yf
import matplotlib.pyplot as plt  
import statsmodels.api as sm, pandas as pd, statistics

from statsmodels.tsa.statespace.tools import diff

from statsmodels.tsa.arima_model import ARIMA

def differencing():
    ticker = yf.Ticker("AMZN")
    hist = ticker.history(start="2020-07-01", end="2020-10-1")
    ts = pd.Series(hist["Close"])
    differenced = diff(ts, k_diff=1)
    res = differenced/statistics.stdev(differenced)
    plt.plot(res)
    #sm.qqplot(res, line ='45') 
    #plt.acorr(res, maxlags=30)
    plt.show()

def predict():
    ticker = yf.Ticker("AMZN")
    hist = ticker.history(start="2020-07-01", end="2020-10-1")
    N, t, w = len(hist), len(hist), 7
    ts = pd.Series(hist["Close"])
    ts = ts.astype(float)
    model = ARIMA(ts[0:t], order=(0,1,0))
    res = model.fit()
    print(res.summary())
    res.plot_predict(start=1, end=N+w, alpha=0.05)
    print("Std residuals: "+str(statistics.stdev(res.resid)))
    plt.show()

#differencing()
predict()