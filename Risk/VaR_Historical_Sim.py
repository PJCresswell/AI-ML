import pandas as pd
import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# VaR Historical simulation: bootstrapping from historical data

# Maximum possible loss during time T excluding outcomes whose probability is less than confidence level alpha
# under normal market conditions

# Obtaining historical returns for the Nikkei 225 Index
data = yf.Ticker('^N225').history(period='1y')
print(data.head())
returns = np.diff(np.log(data['Close']))
plt.plot(returns)
plt.show()

# Simulate returns over the next 5 days
num_days = 5
simulated_returns = np.random.choice(returns, size=num_days, replace=True)
print(simulated_returns)
# Simulate prices
initial_price = 100.
prices = initial_price * np.exp(np.cumsum(simulated_returns))
plt.plot(prices)
plt.show()

def final_price():
    num_days = 20
    simulated_returns = np.random.choice(returns, size=num_days, replace=True)
    prices = initial_price * np.exp(np.cumsum(simulated_returns))
    return prices[-1]

num_samples = 100000
prices = [final_price() for i in range(num_samples)]
plt.hist(prices, bins=100)
plt.show()

def profit_and_loss(final_price):
    return final_price - initial_price

p_and_l = np.vectorize(profit_and_loss)(prices)
plt.hist(p_and_l, bins=100)
plt.show()

for p in range(1, 6):
    print("Quantile %.2f is %.4f" % (p/100., np.percentile(p_and_l, q=p)))
var = -1 * np.percentile(p_and_l, q=5)
print("5%%-VaR is %.4f" % var)