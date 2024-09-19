import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import signal
from typing import List
from pandas import Series

# https://medium.com/@crisvelasquez/riding-the-waves-of-stock-prices-with-wavelet-transform-signals-in-python-e2e81217f9fd


# # Download ASML.AS historical stock data
# ticker = "SIE.DE"
# stock_data = yf.download(ticker, start="2018-01-01", end="2023-12-30")


def compute_cwt(stock_data) -> (Series, Series):
    # Compute the Continuous Wavelet Transform (CWT) using the Ricker wavelet
    widths = np.arange(1, 15)
    cwt_result = signal.cwt(stock_data['Close'].values, signal.ricker, widths)

    # Extract the relevant CWT coefficients for analysis
    cwt_positive = np.where(cwt_result > 0, cwt_result, 0)
    cwt_negative = np.where(cwt_result < 0, cwt_result, 0)

    # Calculate the buy and sell signals from the CWT coefficients
    buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
    sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)

    # Identify buy and sell signal crossovers
    cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
    cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))

    return cross_above, cross_below


# # Define parameter ranges
# widths_range = np.arange(1, 15)
# threshold_range = np.arange(0.1, 1.0, 0.1)

# # Store the best parameters and their performance
# best_widths = None
# best_threshold = None
# best_performance = -np.inf


def backtest(stock_data, cross_above: List[bool], cross_below: List[bool]):
    # We'll start with one "unit" of cash
    cash = 1.0
    stock = 0.0
    position = "out"

    # Go through each day in our data
    for i in range(len(stock_data)):
        # If we have a buy signal and we're not already holding the stock
        if cross_above[i] and position == "out":
            # Buy the stock
            stock += cash / stock_data['Close'][i]
            cash = 0.0
            position = "in"
        # If we have a sell signal and we're holding the stock
        elif cross_below[i] and position == "in":
            # Sell the stock
            cash += stock * stock_data['Close'][i]
            stock = 0.0
            position = "out"

    # Return our final portfolio value
    if position == "in":
        return cash + stock * stock_data['Close'][-1]
    else:
        return cash


def find_best_parameters(stock_data, widths_range = np.arange(1, 15), threshold_range = np.arange(0.1, 1.0, 0.1)):
    # Store the best parameters and their performance
    best_widths = None
    best_threshold = None
    best_performance = -np.inf

    # Go through each combination of parameters
    for widths in widths_range:
        for threshold in threshold_range:
            # Compute the CWT
            cwt_result = signal.cwt(stock_data['Close'].values, signal.ricker, [widths])

            # Extract relevant coefficients
            cwt_positive = np.where(cwt_result > threshold, cwt_result, 0)
            cwt_negative = np.where(cwt_result < -threshold, cwt_result, 0)

            # Calculate signals
            buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
            sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)
            cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
            cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))

            # Calculate performance based on trading signals
            performance = backtest(stock_data, cross_above, cross_below)

            # Update best parameters if this performance is better
            if performance > best_performance:
                best_performance = performance
                best_widths = widths
                best_threshold = threshold

    # Print the best parameters
    print(f"Best widths: {best_widths}")
    print(f"Best threshold: {best_threshold}")
    print(f"Best performance: {best_performance * 100}%")

    # Recalculate the CWT and buy/sell signals using the best parameters
    cwt_result = signal.cwt(stock_data['Close'].values, signal.ricker, [best_widths])
    cwt_positive = np.where(cwt_result > best_threshold, cwt_result, 0)
    cwt_negative = np.where(cwt_result < -best_threshold, cwt_result, 0)
    buy_signal = pd.Series(np.sum(cwt_positive, axis=0), index=stock_data.index)
    sell_signal = pd.Series(-np.sum(cwt_negative, axis=0), index=stock_data.index)
    cross_above = (buy_signal >= sell_signal) & (buy_signal.shift(1) < sell_signal.shift(1))
    cross_below = (buy_signal <= sell_signal) & (buy_signal.shift(1) > sell_signal.shift(1))

    return cross_above, cross_below


def plot_buy_sell(ticker, stock_data, cross_above, cross_below):
    # Plot the stock prices and buy/sell signals
    plt.figure(figsize=(30, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Prices', alpha=0.5)
    plt.scatter(stock_data.index[cross_above], stock_data['Close'][cross_above], label='Buy Signal', marker='^', color='g')
    plt.scatter(stock_data.index[cross_below], stock_data['Close'][cross_below], label='Sell Signal', marker='v', color='r')
    plt.title(f'{ticker} Historical Close Prices with Wavelet Transform Buy and Sell Signals')
    plt.legend()
    plt.show()

# def plot_buy_sell(stock_data):
#     # Plot the stock prices and buy/sell signals
#     plt.figure(figsize=(30, 6))
#     plt.plot(stock_data.index, stock_data['Close'], label='Close Prices', alpha=0.5)
#     plt.scatter(stock_data.index[cross_above], stock_data['Close'][cross_above], label='Buy Signal', marker='^', color='g')
#     plt.scatter(stock_data.index[cross_below], stock_data['Close'][cross_below], label='Sell Signal', marker='v', color='r')
#     plt.title(f'{ticker} Historical Close Prices with Wavelet Transform Buy and Sell Signals')
#     plt.legend()
#     plt.show()




"""
2. Wavelet Transform Theory

At its core, a wavelet is a brief oscillation that has its energy concentrated in time, ensuring it’s both
short-lived and limited in duration. Imagine striking a drum: the sound produced is intense but fades quickly.
This ephemeral characteristic of wavelets makes them immensely suitable to analyze financial signals that are
non-stationary (i.e., their statistical properties change over time). Wavelet Transform is akin to providing
a microscope to view the intricate details of stock data, capturing both its large trends and minute fluctuations.

2.1 Why Not Fourier Transform?
Many might think, why not just use the Fourier Transform, which is a very popular tool to analyze signals?
Fourier Transform breaks down a signal into its constituent sinusoids. However, its Achilles’ heel is its
inability to provide both time and frequency information simultaneously. While it can tell us the frequencies
present, it’s often oblivious to when they occur. Unlike its Fourier counterpart, Wavelet Transform captures
both the frequency and the time, providing a time-frequency representation of the signal.

2.2 The Essence of Wavelet Transform
Mathematically, the Wavelet Transform can be represented as:

Where:
W is the result of our comparison, giving us a measure of similarity.
f (t) represents our stock data.
ψ (t) is the chosen wavelet pattern.

In essence, we’re moving our wavelet ψ across the stock data f(t), looking for places where they align well.
This gives us the wavelet coefficients W(b) which tell us how strongly the stock data matches the wavelet at
each point in time. Put simply, the better the match at any given point, the higher the value of W.

4. Interpretation and Analysis

After implementing the wavelet transform on stock data and refining it with optimized parameters, we arrive at a
series of buy and sell signals. Upon a closer look:

Sensitivity to Market Noise: Just like other technical indicators, the wavelet transform isn’t immune to market
noise. However, its multi-scale nature provides a degree of resilience, allowing it to highlight genuine patterns
 while potentially filtering out short-term fluctuations.

Periodic Patterns: Stocks often exhibit cyclical behaviors — quarterly financial reports, yearly market trends,
etc. The wavelet’s ability to capture such behaviors at different scales can be a significant advantage in
predicting future movements.

Performance Metrics: While our optimization boosted historical performance, it’s essential to validate the
strategy on out-of-sample data. Performance on historical data doesn’t guarantee similar results in the future.
It’s crucial to be aware of potential overfitting.

5. Potential Improvements and Extensions

Alternative Wavelets: We utilized the Ricker wavelet, but several other wavelet families, such as Daubechies or Haar, might offer different insights. Exploring these can be an exciting avenue.

Incorporating Other Indicators: While the wavelet transform provides a unique perspective, combining it with other technical indicators might strengthen the decision-making process.

Machine Learning Integration: Leveraging machine learning algorithms can help in predicting buy/sell signals more accurately. The wavelet coefficients can serve as features for models like decision trees, neural networks, etc.

Adaptive Thresholding: Instead of a static threshold, dynamic thresholding methods that adapt to changing market conditions might improve signal reliability.
"""

