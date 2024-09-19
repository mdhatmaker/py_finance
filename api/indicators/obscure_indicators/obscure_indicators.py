import yfinance as yf
import pandas as pd
import ta
import numpy as np
from datetime import datetime
from pandas import Series
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc

# https://medium.com/@crisvelasquez/8-trading-indicators-in-python-you-never-heard-of-4d3a043dda95


def download_stock_data(ticker_symbol: str, startDate: str = '2020-01-01', endDate: str = None) -> Series:
    """Download historical stock data."""
    #ticker_symbol = "SAP.DE"
    if not endDate:
        endDate = datetime.now()
    print(f'{ticker_symbol}    {startDate} to {endDate}')
    stock_data = yf.download(ticker_symbol, start=startDate, end=endDate)
    return stock_data


def calculate_true_range(data):
    """Calculate the True Range of the stock data."""
    return np.maximum.reduce([
        data['High'] - data['Low'],
        abs(data['High'] - data['Close'].shift()),
        abs(data['Low'] - data['Close'].shift())
    ])


def calculate_indicators(data, window_size):
    """Calculate ATR, highest high, lowest low, and Choppiness Index."""
    data['TR'] = calculate_true_range(data)
    data['ATR'] = data['TR'].rolling(window=window_size).mean()
    data['highestHigh'] = data['High'].rolling(window=window_size).max()
    data['lowestLow'] = data['Low'].rolling(window=window_size).min()

    # Choppiness Index
    data['Sum_TR'] = data['TR'].rolling(window=window_size).sum()
    data['Range'] = data['highestHigh'] - data['lowestLow']
    data['CHOP'] = 100 * np.log10(data['Sum_TR'] / data['Range']) / np.log10(window_size)
    data['CHOP'] = data['CHOP'].clip(lower=0, upper=100)

    # Awesome Oscillator
    data['ao'] = ta.momentum.AwesomeOscillatorIndicator(data['High'], data['Low']).awesome_oscillator()
    data['signal_line'] = ta.trend.ema_indicator(data['ao'], window=9)

    # MACD
    macd_indicator = ta.trend.MACD(data['Close'])
    data['macd'] = macd_indicator.macd()
    data['macd_signal'] = macd_indicator.macd_signal()

    return data


def generate_ci_signals(data):
    """Generate buy and sell signals based on Choppiness Index."""
    data['CHOP_lag1'] = data['CHOP'].shift()
    data['signal'] = np.where((data['CHOP'] < 30) & (data['CHOP_lag1'] >= 30), 'look for buy signals',
                              np.where((data['CHOP'] > 60) & (data['CHOP_lag1'] <= 60), 'look for sell signals', 'neutral'))

    return data


def plot_ci_data(ticker, data):
    """Plot stock prices and signals."""
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax[0].plot(data['Close'], label='Close price', color='blue')
    buy_signals = data[data['signal'] == 'look for buy signals']
    sell_signals = data[data['signal'] == 'look for sell signals']
    ax[0].scatter(buy_signals.index, buy_signals['Close'], color='green', label='Potential Buy Signal', marker='^', alpha=1, s=100)
    ax[0].scatter(sell_signals.index, sell_signals['Close'], color='red', label='Potential Sell Signal', marker='v', alpha=1, s=100)
    ax[0].set_title(f"{ticker} Close Price")
    ax[0].legend()

    ax[1].plot(data['CHOP'], label='Choppiness Index', color='purple')
    ax[1].axhline(60, color='red', linestyle='--', label='Sell Threshold')
    ax[1].axhline(30, color='green', linestyle='--', label='Buy Threshold')
    ax[1].set_title(f"{ticker} Choppiness Index")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Function to calculate disparity index
def get_di(data, lookback):
    ma = data.rolling(lookback).mean()
    return ((data - ma) / ma) * 100


# Implement the DI strategy
def implement_di_strategy(prices, di):
    buy_price = []
    sell_price = []
    di_signal = []
    signal = 0

    for i in range(len(prices)):
        if di[i - 4] < 0 and di[i - 3] < 0 and di[i - 2] < 0 and di[i - 1] < 0 and di[i] > 0:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                di_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                di_signal.append(0)
        elif di[i - 4] > 0 and di[i - 3] > 0 and di[i - 2] > 0 and di[i - 1] > 0 and di[i] < 0:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                di_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                di_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            di_signal.append(0)

    return buy_price, sell_price, di_signal


# Function to execute the Disparity Index strategy
def execute_di_strategy(stock_symbol, stock_data):      #stock_symbol, start_date, end_date):
    #stock_data = download_stock_data(stock_symbol, start_date, end_date)
    stock_data['di_14'] = get_di(stock_data['Close'], 14)
    stock_data.dropna(inplace=True)

    buy_price, sell_price, _ = implement_di_strategy(stock_data['Close'], stock_data['di_14'])

    # Plotting the buy and sell signals along with DI
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Plotting the stock price and signals
    ax[0].plot(stock_data['Close'], label='Close Price', alpha=0.5)
    ax[0].scatter(stock_data.index, buy_price, label='Buy Signal', marker='^', color='green', s=100)
    ax[0].scatter(stock_data.index, sell_price, label='Sell Signal', marker='v', color='red', s=100)
    ax[0].set_title(f'{stock_symbol} - Buy & Sell Signals')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Plotting the Disparity Index with bars
    ax[1].bar(stock_data.index, stock_data['di_14'], color=np.where(stock_data['di_14'] >= 0, '#26a69a', '#ef5350'))
    ax[1].axhline(0, color='gray', linestyle='--')  # Add a line at zero
    ax[1].set_title(f'{stock_symbol} - 14-Day Disparity Index')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Disparity Index (%)')

    plt.tight_layout()
    plt.show()


# def calculate_indicators(data):
#     """Calculate trading indicators."""
#     # Awesome Oscillator
#     data['ao'] = ta.momentum.AwesomeOscillatorIndicator(data['High'], data['Low']).awesome_oscillator()
#     data['signal_line'] = ta.trend.ema_indicator(data['ao'], window=9)
#
#     # MACD
#     macd_indicator = ta.trend.MACD(data['Close'])
#     data['macd'] = macd_indicator.macd()
#     data['macd_signal'] = macd_indicator.macd_signal()
#
#     return data


def generate_ao_signals(data):
    """Generate trading signals for Awesome Oscillator."""
    data['zero_cross'] = np.where((data['ao'].shift(1) < 0) & (data['ao'] > 0), True,
                                  np.where((data['ao'].shift(1) > 0) & (data['ao'] < 0), False, np.NaN))
    return data


def generate_macd_signals(data):
    """Generate trading signals for MACD."""
    data['macd_cross'] = np.where((data['macd'].shift(1) < data['macd_signal'].shift(1)) & (data['macd'] > data['macd_signal']), True,
                                  np.where((data['macd'].shift(1) > data['macd_signal'].shift(1)) & (data['macd'] < data['macd_signal']), False, np.NaN))
    return data


# Plot Awesome Oscillator
def plot_ao_data(ticker, data):
    """Plot stock prices and indicators."""
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15, 8))

    # Plot stock price with signals
    ax1.plot(data.index, data['Close'], label=f'{ticker}')
    buy_signals = data[data['zero_cross'] == True]
    sell_signals = data[data['zero_cross'] == False]
    ax1.plot(buy_signals.index, data.loc[buy_signals.index]['Close'], '^', markersize=10, color='g', label='buy')
    ax1.plot(sell_signals.index, data.loc[sell_signals.index]['Close'], 'v', markersize=10, color='r', label='sell')

    macd_cross_buy = data[data['macd_cross'] == True]
    macd_cross_sell = data[data['macd_cross'] == False]
    ax1.plot(macd_cross_buy.index, data.loc[macd_cross_buy.index]['Close'], 'o', markersize=7, color='purple', label='MACD Cross Buy')
    ax1.plot(macd_cross_sell.index, data.loc[macd_cross_sell.index]['Close'], 'o', markersize=7, color='brown', label='MACD Cross Sell')
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_ylabel('Price (€)')
    ax1.legend(loc='upper left')

    # Plot Awesome Oscillator histogram
    ax2.bar(data.index, data['ao'] - data['signal_line'], color=['g' if data['ao'].iloc[i] > data['signal_line'].iloc[i] else 'r' for i in range(len(data))], label='Awesome Oscillator')
    ax2.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.7)
    ax2.set_title('Awesome Oscillator')
    ax2.set_ylabel('AO')
    ax2.plot(data.index, data['signal_line'], label='Signal Line', color='orange')
    ax2.plot(data.index, data['macd'], label='MACD', color='green')
    ax2.legend(loc='best')

    plt.show()





def run_obscure_indicators(ticker: str, start_date, end_date):
    data = download_stock_data(ticker, start_date, end_date)

    # Execute Choppiness Index
    window_size = 14  # You can adjust this value
    ind_data = calculate_indicators(data, window_size)

    sig = generate_ci_signals(ind_data)
    plot_ci_data(ticker, sig)

    # Execute Disparity Index
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    execute_di_strategy(ticker, data)       #start_date, end_date)

    # Execute Awesome Oscillator
    #ind = calculate_indicators(data)
    sig = generate_ao_signals(ind_data)
    sig = generate_macd_signals(sig)
    plot_ao_data(ticker, sig)






"""
The Choppiness Index quantifies the degree of market volatility.
It’s calculated using a logarithmic formula that compares the sum of the True Range over a set number of
periods (N) to the range of the market’s highs and lows over the same period.

Traders use the Choppiness Index to distinguish between trending and range-bound markets.
For instance, a low Choppiness Index value signals a strong trend, while high values indicate a more directionless market.
This information helps traders decide whether to employ trend-following strategies or focus on range trading tactics.
"""

"""
The Disparity Index is a technical indicator that measures the percentage difference between the latest closing
price and a chosen moving average, reflecting short-term price fluctuations relative to a longer trend.

Traders use this index to identify potential price reversals. A significant deviation from the moving average,
indicated by a high Disparity Index value, often precedes a reversion to the mean.

For example, a threshold of +/- 5% can signal when to consider entering or exiting a trade. Positive values
indicate that the price is above the moving average (potential overvaluation), and negative values indicate
the price is below the moving average (potential undervaluation).
"""

"""
The Awesome Oscillator (AO) and Moving Average Convergence Divergence (MACD) are both momentum indicators, but
they calculate momentum differently.

While the MACD is effective in trend confirmation, the Awesome Oscillator can more quickly respond to immediate
price changes. Combining these, a trader might use the MACD for trend direction and the Awesome Oscillator for
precise entry and exit points.

The Awesome Oscillator is known for its ability to capture market momentum in the short term, whereas the MACD
is often used for identifying longer-term trend direction and momentum changes.
"""



"""
3. Applications to Trading

Trend Confirmation: Technical indicators such as moving averages confirm market trends and can signal changes in
momentum, aiding in the decision to enter or exit trades.

Momentum Measurement: Indicators like the Awesome Oscillator measure the velocity of price changes, helping to
anticipate continuations or reversals in price trends.

Market Sentiment: Tools like the RVI assess buying or selling pressure, providing insights into market sentiment
and potential turning points.

Overbought/Oversold Identification: Indicators such as the DeMarker highlight extreme price conditions, which
may indicate impending reversals or continuations.
"""


