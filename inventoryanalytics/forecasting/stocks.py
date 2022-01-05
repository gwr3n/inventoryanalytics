'''
inventoryanalytics: a Python library for Inventory Analytics

Author: Roberto Rossi

MIT License
  
Copyright (c) 2018 Roberto Rossi
'''

import datetime
import numpy as np
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt  
import statsmodels.api as sm, pandas as pd, statistics

from statsmodels.tsa.statespace.tools import diff

from statsmodels.tsa.arima_model import ARIMA

def differencing():
    ticker = yf.Ticker("AMZN")
    hist = ticker.history(start="2020-09-01", end="2020-10-3")
    ts = pd.Series(hist["Close"])
    differenced = diff(ts, k_diff=1)
    res = differenced/statistics.stdev(differenced)
    plt.plot(res)
    #sm.qqplot(res, line ='45') 
    #plt.acorr(res, maxlags=30)
    plt.show()

def predict(fig, ax):
    prediction_window = 15
    training_window = 60
    symbol = "V"
    ticker = yf.Ticker(symbol)
    now = datetime.datetime.now()# - timedelta(days=15)
    start_window = now - timedelta(days=training_window+prediction_window)
    print(now)
    print(start_window)

    hist = ticker.history(start=start_window.strftime("%Y-%m-%d"), end=now.strftime("%Y-%m-%d"))
    #print(hist)
    t, w = len(hist)-prediction_window, prediction_window
    
    #ts = pd.Series(hist["Close"].values)
    #print(np.append(hist["Close"].values[1:], yf.download(tickers=symbol, period='1d', interval='5m').tail(1)["Close"].values[0]))
    ts = pd.Series(np.append(hist["Close"].values[1:], yf.download(tickers=symbol, period='1d', interval='5m').tail(1)["Close"].values[0]))
    ts = ts.astype(float)
    model = ARIMA(ts[0:t], order=(0,1,0))
    res = model.fit()
    print(res.summary())

    print(t)
    res.plot_predict(start=2, end=t+w, alpha=0.05, ax=ax)
    plt.plot(ts[t-1:t+w], label="realisations")
    plt.title(symbol + "    " + start_window.strftime("%Y-%m-%d") + "    " + now.strftime("%Y-%m-%d"))
    print("Std residuals: "+str(statistics.stdev(res.resid)))
    plt.show()
    #plt.draw()
    #plt.pause(1)
    #plt.cla()

#differencing()
fig, ax = plt.subplots()
predict(fig, ax)