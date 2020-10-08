import datetime
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt  
import statsmodels.api as sm, pandas as pd, statistics

from statsmodels.tsa.statespace.tools import diff

from statsmodels.tsa.arima_model import ARIMA

def differencing():
    ticker = yf.Ticker("V")
    hist = ticker.history(start="2020-09-01", end="2020-10-3")
    ts = pd.Series(hist["Close"])
    differenced = diff(ts, k_diff=1)
    res = differenced/statistics.stdev(differenced)
    plt.plot(res)
    #sm.qqplot(res, line ='45') 
    #plt.acorr(res, maxlags=30)
    plt.show()

def predict():
    prediction_window = 15
    training_window = 60
    symbol = "AMZN"
    ticker = yf.Ticker(symbol)
    now = datetime.datetime.now()
    start_window = now - timedelta(days=training_window+prediction_window)
    print(now)
    print(start_window)

    hist = ticker.history(start=start_window.strftime("%Y-%m-%d"), end=now.strftime("%Y-%m-%d"))
    t, w = len(hist)-prediction_window, prediction_window
    
    ts = pd.Series(hist["Close"].values)
    ts = ts.astype(float)
    model = ARIMA(ts[0:t], order=(0,1,0))
    res = model.fit()
    print(res.summary())

    fig, ax = plt.subplots()
    print(t)
    res.plot_predict(start=2, end=t+w, alpha=0.05, ax=ax)
    plt.plot(ts[t-1:t+w], label="realisations")
    plt.title(symbol + "    " + start_window.strftime("%Y-%m-%d") + "    " + now.strftime("%Y-%m-%d"))
    print("Std residuals: "+str(statistics.stdev(res.resid)))
    plt.show()

#differencing()
predict()


