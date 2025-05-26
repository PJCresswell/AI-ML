import yfinance as yf
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
# Set seaborn plot style
sns.set_style("darkgrid")
# Set matplotlib plot style
style.use("fivethirtyeight")
plt.rcParams["figure.autolayout"] = True

tickers = ['BAC', 'JPM', 'C']
api = yf.Tickers(tickers)
raw = api.history(period='5y')
market_data = raw['Close']
market_data.index = market_data.index.date
market_data.index.names = ['date']

for company in tickers:
    # Normal returns
    market_data[company + '_Return'] = market_data[company].pct_change() * 100
    # Log returns take into account the compounding effect
    market_data[company + '_LogReturn'] = np.log(market_data[company] / market_data[company].shift(1))
market_data.dropna(inplace=True)

'''
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(market_data['BAC_Return'].dropna(), color='lightcoral')
ax.set(title='BAC', ylabel='% Return')
plt.show()
'''

# Calculate the constant variance across the period
market_data['BAC_Var'] = market_data['BAC_Return'].var()

# Now we look at a model for time varying variance / volatility
# https://gist.github.com/hrishipoola/d96e4d6bc0b525231703541a49262216

from arch import arch_model

# GARCH
# Variance(t) = weight + Alpha x residual(t-1)squared + Beta x Variance(t-1)
# Alpha represents how volatility reacts to new information. The larger the alpha the larger the immediate impact expressed as residuals
# The larger the Beta the longer the duration of the impact
# GARCH(1,1) states that variance of time t equals a constant, omega, plus alpha x residual squared of time t-1, plus beta x variance of time t-1

# Assumptions : Distribution of residuals = normal; Mean model = constant
# Specify and fit the model
basic_gm = arch_model(market_data['BAC_Return'], p=1, q=1, mean='constant', vol='GARCH', dist='normal')
gm_result = basic_gm.fit()
# Print out the results - the coefficients of the mean and volatility models
print(gm_result.summary())
# Plots : Standardized residuals :
# Plots : Conditional volatility :
gm_result.plot()
plt.show()

# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5)

# Print the forecast variance
print(gm_forecast.variance[-1:])

# Calculate standardized residuals
gm_std_resid = gm_result.resid / gm_result.conditional_volatility

# Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(gm_std_resid, color='salmon', bins=40)
ax.set(title='Distribution of Standardized Residuals')
plt.show()

# Get model estimated volatility
normal_volatility = gm_result.conditional_volatility

# Plot model fitting results
plt.figure(figsize=(12,6))
plt.plot(normal_volatility, color = 'turquoise', label = 'Normal Volatility')
plt.plot(market_data['BAC_Return'], color = 'grey', label = 'Daily Returns', alpha = 0.4)
plt.legend(loc = 'upper right', frameon=False)
plt.show()
