import numpy as np
import pandas as pd
import requests
import yfinance as yf

confidence_level = 0.99

# VaR of the bond positions
# Load in the historical data
df = pd.read_csv('data\BondData.csv')
print(df.head())
exit()

# Load in our portfolio
df = pd.read_csv('data\Portfolio.csv')
print(df.head())

# VaR of the stock positions
#stock = yf.Ticker('TSLA')   # -0.0844
#stock = yf.Ticker('MSFT')   # -0.0367
stock = yf.Ticker('BAC')    # -0.0315
stock_his = stock.history(period="1y")
stock_rtns = stock_his['Close'].pct_change().dropna()
stock_VaR = np.percentile(stock_rtns, (1 - confidence_level) * 100)
print(stock_VaR)
# Then need to multiply VaR by the position value = share price * amount held

# VaR of the futures positions
#Fut_his = yf.download("ZB=F", period="1y")  # US Treasury Bond Futures for March      -0.0163
Fut_his = yf.download("ES=F", period="1y") # S&P500 Index Futures for March     -0.0218
Fut_rtns = Fut_his['Adj Close'].pct_change().dropna()
Fut_VaR = np.percentile(Fut_rtns, (1 - confidence_level) * 100)
print(Fut_VaR)
# Then need to multiply VaR by the position value = share price * amount held



# VaR of the option positions

url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=TSLA&date=2025-02-03&apikey=I2UR68ST032EG0J5&datatype=csv'
AD = pd.read_csv(url)
print(AD)
AD.to_csv('data/testfile.csv')







