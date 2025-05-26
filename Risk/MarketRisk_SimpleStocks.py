import yfinance as yf
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
import matplotlib.pyplot as plt

def show_returns(df):
    # Plot out the normal returns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.set_title('BAC Returns')
    ax1.plot(df['BAC_Return'], 'tab:orange')
    ax2.set_title('JPM Returns')
    ax2.plot(df['JPM_Return'], 'tab:green')
    ax3.set_title('Citi Returns')
    ax3.plot(df['C_Return'], 'tab:red')
    plt.show()

def main():
    # Load in EOD prices for the three stocks
    tickers = ['BAC', 'JPM', 'C']
    api = yf.Tickers(tickers)
    raw = api.history(period='5y')
    market_data = raw['Close']
    market_data.index = market_data.index.date
    market_data.index.names = ['date']

    for company in tickers:
        # Calculate normal returns
        market_data[company + '_Return'] = market_data[company].pct_change()
        # Calculate log returns : takes into account the compounding effect
        market_data[company + '_LogReturn'] = np.log(market_data[company] / market_data[company].shift(1))

        # Calculate the mean, variance and standard deviation of returns across a 3 month rolling window
        # The standard deviation is the square root of the variance
        # Calculate skewness - How symmetrical the distribution is around the mean. Investing in portfolios with negative skewness = riskier
        # Calculate kurtosis - how spread out the results are. More weight put on extreme point. Higher kurtosis = riskier
        market_data[company + '_Mean'] = market_data[company + '_Return'].rolling(90).mean()
        market_data[company + '_Var'] = market_data[company + '_Return'].rolling(90).var()
        market_data[company + '_Std'] = market_data[company + '_Return'].rolling(90).std()
        market_data[company + '_Skew'] = market_data[company + '_Return'].rolling(90).skew()
        market_data[company + '_Kurt'] = market_data[company + '_Return'].rolling(90).kurt()

        # Next we weight the more recent observations. Weights add up to one. Half life = when half the weight allocate
        market_data[company + '_EWMean'] = market_data[company + '_Return'].ewm(halflife=30).mean()
        market_data[company + '_EWVar'] = market_data[company + '_Return'].ewm(halflife=30).var()
        market_data[company + '_EWStd'] = market_data[company + '_Return'].ewm(halflife=30).std()

    market_data.dropna(inplace=True)
    # print(market_data.tail())
    market_data.to_csv('Simple_Stock_Data_Extract.csv')

    # Used by the historical & hybrid VaR calculations
    historical_series = market_data[-256:]
    historical_series['row_number'] = historical_series.reset_index().index
    historical_series['HistWeight'] = 0.004
    historical_series['HybridWeight'] = (0.98 ** (255 - historical_series['row_number']))
    total_weight = historical_series['HybridWeight'].sum()
    historical_series['HybridWeightPercent'] = historical_series['HybridWeight'] / total_weight

    for company in tickers:

        # Method 1 : Measure the market risk using standard deviation
        print(company + ' : Standard Deviation : {:.3f}'.format(market_data[company+ '_Std'][-1]))
        # Now work out the dollar standard deviation
        dollar_value = market_data[company+ '_Std'][-1] * market_data[company][-1]
        print(company + ' : Dollar Standard Deviation : {:.3f}'.format(dollar_value))

        # Method 2 : Measure the market risk using standard deviation with decay
        print(company + ' : Standard Deviation with decay : {:.3f}'.format(market_data[company + '_EWStd'][-1]))
        # Now work out the dollar standard deviation
        dollar_value = market_data[company + '_EWStd'][-1] * market_data[company][-1]
        print(company + ' : Dollar Standard Deviation with decay : {:.3f}'.format(dollar_value))

        # Display the skewness and kurtosis
        print(company + ' : Skewness : Negative = riskier : {:.3f}'.format(market_data[company + '_Skew'][-1]))
        print(company + ' : Kurtosis : Higher = riskier : {:.3f}'.format(market_data[company + '_Kurt'][-1]))

        # Method 3 : Delta normal VaR. StdDev with decay multiplied by current price then by factor
        # -1.64 is the appropriate factor for the inverse of the standard normal distribution for a 95% VaR. In Excel NORM.S.INV(0.05) = -1.64
        pos_std = market_data[company + '_EWStd'][-1] * market_data[company][-1]
        delta_normal = pos_std * -1.64
        print(company + ' : Delta Normal VaR : {:.3f}'.format(delta_normal))

        # Methods 4 & 5 : Calculate Historical and Hybrid VaR using the last 256 days of data
        historical_series.sort_values(by=company + '_Return', inplace=True)
        historical_series[company + '_HistCumSum'] = historical_series['HistWeight'].cumsum()
        historical_series[company + '_HybridCumSum'] = historical_series['HybridWeightPercent'].cumsum()
        #historical_series['row_number'] = historical_series.reset_index().index
        #historical_series[company+'_HistCumWeight'] = (historical_series['row_number'] + 1) * (1/256)

        # Historical VaR
        histVaRrow = historical_series.iloc[np.argmax(historical_series[company+'_HistCumSum'] > 0.05)]
        HistVaR = histVaRrow[company+'_Return'] * market_data[company][-1]
        print(company + ' : Historical VaR : {:.3f} x {:.3f} = {:.3f}'.format(histVaRrow[company+'_Return'], market_data[company][-1], HistVaR))
        # Hybrid VaR
        hybridVaRrow = historical_series.iloc[np.argmax(historical_series[company + '_HybridCumSum'] > 0.05)]
        HybridVaR = hybridVaRrow[company + '_Return'] * market_data[company][-1]
        print(company + ' : Hybrid VaR : {:.3f} x {:.3f} = {:.3f}'.format(hybridVaRrow[company + '_Return'], market_data[company][-1], HybridVaR))
        historical_series.to_csv('Test_Extract.csv')

        # Method 6 : Calculate Expected Shortfall
        # If we are measuring VaR at the 95% confidence level, the expected Shortfall would be the average loss in the 5% cases where the fund exceeds its VaR
        HistES = historical_series[company+'_Return'][:12].mean() * market_data[company][-1]
        print(company + ' : Expected Shortfall : {:.3f} x {:.3f} = {:.3f}'.format(historical_series[company+'_Return'][:12].mean(), market_data[company][-1],HistES))

if __name__ == "__main__":
    main()