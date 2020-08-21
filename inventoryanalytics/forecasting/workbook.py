# https://pypi.org/project/yfinance/
import yfinance as yf
import matplotlib.pyplot as plt  

msft = yf.Ticker("GOOGL")
hist = msft.history(start="2020-01-01", end="2020-08-20")
plt.xlabel('Period')
plt.ylabel('Value')
plt.plot(hist['Close'], "r", label="GOOGL")
plt.show()