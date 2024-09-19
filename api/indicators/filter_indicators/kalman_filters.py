from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas import Series, DataFrame
from pykalman import KalmanFilter
from itertools import product
from typing import List, Dict

from api.utils.yahoo_finance import download_close_prices, download_ohlc

# Kalman filters docs
# https://medium.com/@serdarilarslan/implementing-a-kalman-filter-based-trading-strategy-8dec764d738e
# https://quantitativepy.substack.com/p/implementing-a-kalman-filter-based
"""
Strategy Rules:
Long Entry: Enter a long position when the z-score is below a certain threshold.
Short Entry: Enter a short position when the z-score is above a certain threshold.
Exit Long: Exit a long position when the z-score reverts to a higher threshold.
Exit Short: Exit a short position when the z-score reverts to a lower threshold.
Stop-Loss: Exit the position if the price moves beyond a specified percentage from the entry price.
"""


# def download_ohlc(symbol: str, startDate: str = '2018-01-01', endDate: str = None) -> DataFrame:
#     if not endDate:
#         endDate = datetime.now().strftime('%Y-%m-%d')
#     # Download data for a single ticker symbol, e.g., AAPL
#     print(f'{symbol}    {startDate} to {endDate}')
#     # df = yf.download('AAPL', start='2020-05-22', end='2024-05-22')
#     df = yf.download(symbol, start=startDate, end=endDate)
#     return df
#
#
# def download_close_prices(symbol: str, startDate: str = '2018-01-01', endDate: str = None) -> Series:
#     if not endDate:
#         endDate = datetime.now().strftime('%Y-%m-%d')
#     # Download data for a single ticker symbol, e.g., AAPL
#     print(f'{symbol}    {startDate} to {endDate}')
#     data = download_ohlc(symbol, startDate, endDate)
#     close_prices = data['Close']
#     return close_prices


def calc_kalman_filter(close_prices: Series):
    # Initialize Kalman filter parameters for the moving average estimation
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    # Use the Kalman filter to estimate the moving average
    state_means, state_covariances = kf.filter(close_prices.values)
    kalman_avg = state_means.flatten()

    # Calculate the spread and z-score using the Kalman filter estimate
    window = 30
    spread = close_prices - kalman_avg
    rolling_std = spread.rolling(window=window).std()
    zscore = spread / rolling_std
    return kalman_avg, spread, zscore


# Function to backtest the strategy with given thresholds
def backtest_strategy(close_prices: Series, kalman_avg, spread, zscore,
                      long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold, stop_loss_threshold):
    signals = pd.DataFrame(index=close_prices.index)
    signals['price'] = close_prices
    signals['kalman_avg'] = kalman_avg
    signals['spread'] = spread
    signals['zscore'] = zscore
    signals['long'] = (signals['zscore'] <= long_entry_threshold)
    signals['short'] = (signals['zscore'] >= short_entry_threshold)
    signals['exit_long'] = (signals['zscore'] >= long_exit_threshold)
    signals['exit_short'] = (signals['zscore'] <= short_exit_threshold)
    signals['position'] = 0
    signals['entry_price'] = np.nan

    # Iterate over rows to update positions and apply stop-loss
    for i in range(1, len(signals)):
        if signals['long'].iat[i]:
            signals.at[signals.index[i], 'position'] = 1
            signals.at[signals.index[i], 'entry_price'] = signals['price'].iat[i]
        elif signals['short'].iat[i]:
            signals.at[signals.index[i], 'position'] = -1
            signals.at[signals.index[i], 'entry_price'] = signals['price'].iat[i]
        elif signals['exit_long'].iat[i] and signals['position'].iat[i-1] == 1:
            signals.at[signals.index[i], 'position'] = 0
        elif signals['exit_short'].iat[i] and signals['position'].iat[i-1] == -1:
            signals.at[signals.index[i], 'position'] = 0
        elif signals['position'].iat[i-1] == 1 and signals['price'].iat[i] < signals['entry_price'].iat[i-1] * (1 - stop_loss_threshold):
            signals.at[signals.index[i], 'position'] = 0  # Stop-loss for long position
        elif signals['position'].iat[i-1] == -1 and signals['price'].iat[i] > signals['entry_price'].iat[i-1] * (1 + stop_loss_threshold):
            signals.at[signals.index[i], 'position'] = 0  # Stop-loss for short position
        else:
            signals.at[signals.index[i], 'position'] = signals['position'].iat[i-1]
            signals.at[signals.index[i], 'entry_price'] = signals['entry_price'].iat[i-1]

    returns = close_prices.pct_change().dropna()
    transaction_cost = 0.001  # 0.1% transaction cost
    signals['strategy_returns'] = returns * signals['position'].shift(1)
    signals['transaction_costs'] = transaction_cost * signals['position'].diff().abs()
    signals['net_returns'] = signals['strategy_returns'] - signals['transaction_costs']
    signals['cumulative_returns'] = (1 + signals['net_returns']).cumprod()

    if signals['net_returns'].std() == 0 or np.isnan(signals['net_returns'].mean()):
        sharpe_ratio = -np.inf
    else:
        sharpe_ratio = signals['net_returns'].mean() / signals['net_returns'].std() * np.sqrt(252)

    max_drawdown = (signals['cumulative_returns'].cummax() - signals['cumulative_returns']).max()

    return sharpe_ratio, max_drawdown


def get_best_thresholds(close_prices, kalman_avg, spread, zscore,
                        long_entry_thresholds, short_entry_thresholds, long_exit_thresholds, short_exit_thresholds, stop_loss_thresholds):
    # Initialize variables to store the best performance and thresholds
    best_sharpe_ratio = -np.inf
    best_thresholds = None
    results = []
    count = 0

    # Iterate over all combinations of thresholds
    for long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold, stop_loss_threshold in product(long_entry_thresholds, short_entry_thresholds, long_exit_thresholds, short_exit_thresholds, stop_loss_thresholds):
        if stop_loss_threshold <= 0:
            continue  # Ensure stop-loss thresholds make practical sense
        sharpe_ratio, max_drawdown = backtest_strategy(close_prices, kalman_avg, spread, zscore, long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold, stop_loss_threshold)
        results.append((long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold, stop_loss_threshold, sharpe_ratio, max_drawdown))

        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_thresholds = (long_entry_threshold, short_entry_threshold, long_exit_threshold, short_exit_threshold, stop_loss_threshold)

        count += 1
        if count % 10 == 0:
            print(".", end="")

    # Print the best thresholds and their performance
    print(f"\nBest Thresholds: {best_thresholds}")
    print(f"Best Sharpe Ratio: {best_sharpe_ratio}")

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results, columns=['Long Entry Threshold', 'Short Entry Threshold', 'Long Exit Threshold', 'Short Exit Threshold', 'Stop-Loss Threshold', 'Sharpe Ratio', 'Max Drawdown'])

    return best_sharpe_ratio, best_thresholds, results_df


def plot_cumulative_returns(symbol, close_prices, kalman_avg, spread, zscore, best_thresholds):
    # Plot the cumulative returns for the best thresholds
    best_long_entry_threshold, best_short_entry_threshold, best_long_exit_threshold, best_short_exit_threshold, best_stop_loss_threshold = best_thresholds
    signals = pd.DataFrame(index=close_prices.index)
    signals['price'] = close_prices
    signals['kalman_avg'] = kalman_avg
    signals['spread'] = spread
    signals['zscore'] = zscore
    signals['long'] = (signals['zscore'] <= best_long_entry_threshold)
    signals['short'] = (signals['zscore'] >= best_short_entry_threshold)
    signals['exit_long'] = (signals['zscore'] >= best_long_exit_threshold)
    signals['exit_short'] = (signals['zscore'] <= best_short_exit_threshold)
    signals['position'] = 0
    signals['entry_price'] = np.nan

    # Iterate over rows to update positions and apply stop-loss
    for i in range(1, len(signals)):
        if signals['long'].iat[i]:
            signals.at[signals.index[i], 'position'] = 1
            signals.at[signals.index[i], 'entry_price'] = signals['price'].iat[i]
        elif signals['short'].iat[i]:
            signals.at[signals.index[i], 'position'] = -1
            signals.at[signals.index[i], 'entry_price'] = signals['price'].iat[i]
        elif signals['exit_long'].iat[i] and signals['position'].iat[i-1] == 1:
            signals.at[signals.index[i], 'position'] = 0
        elif signals['exit_short'].iat[i] and signals['position'].iat[i-1] == -1:
            signals.at[signals.index[i], 'position'] = 0
        elif signals['position'].iat[i-1] == 1 and signals['price'].iat[i] < signals['entry_price'].iat[i-1] * (1 - best_stop_loss_threshold):
            signals.at[signals.index[i], 'position'] = 0  # Stop-loss for long position
        elif signals['position'].iat[i-1] == -1 and signals['price'].iat[i] > signals['entry_price'].iat[i-1] * (1 + best_stop_loss_threshold):
            signals.at[signals.index[i], 'position'] = 0  # Stop-loss for short position
        else:
            signals.at[signals.index[i], 'position'] = signals['position'].iat[i-1]
            signals.at[signals.index[i], 'entry_price'] = signals['entry_price'].iat[i-1]

    returns = close_prices.pct_change().dropna()
    transaction_cost = 0.001  # 0.1% transaction cost
    signals['strategy_returns'] = returns * signals['position'].shift(1)
    signals['transaction_costs'] = transaction_cost * signals['position'].diff().abs()
    signals['net_returns'] = signals['strategy_returns'] - signals['transaction_costs']
    signals['cumulative_returns'] = (1 + signals['net_returns']).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    signals['cumulative_returns'].plot()
    plt.title(f'({symbol}) Cumulative Returns of Single Stock Trading Strategy with Optimized Kalman Filter')
    plt.show()

    # Plot the stock price, Kalman moving average, and positions
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices, label='Stock Price')
    plt.plot(signals['kalman_avg'], label='Kalman Filter Moving Average', linestyle='--')
    plt.fill_between(signals.index, close_prices, where=signals['position'] == 1, color='green', alpha=0.2, label='Long')
    plt.fill_between(signals.index, close_prices, where=signals['position'] == -1, color='red', alpha=0.2, label='Short')
    plt.legend()
    plt.title(f'({symbol}) Stock Price and Trading Signals with Optimized Kalman Filter')
    plt.show()

    # Plot the z-score
    plt.figure(figsize=(12, 6))
    plt.plot(zscore, label='Z-score')
    plt.axhline(best_short_entry_threshold, color='red', linestyle='--', label='Short Entry Threshold')
    plt.axhline(best_long_entry_threshold, color='green', linestyle='--', label='Long Entry Threshold')
    plt.axhline(best_long_exit_threshold, color='blue', linestyle='--', label='Long Exit Threshold')
    plt.axhline(best_short_exit_threshold, color='orange', linestyle='--', label='Short Exit Threshold')
    plt.fill_between(signals.index, zscore, where=signals['position'] == 1, color='green', alpha=0.2)
    plt.fill_between(signals.index, zscore, where=signals['position'] == -1, color='red', alpha=0.2)
    plt.legend()
    plt.title(f'({symbol}) Z-score and Trading Signals with Optimized Kalman Filter')
    plt.show()

    # Performance metrics for the optimized strategy
    if signals['net_returns'].std() == 0 or np.isnan(signals['net_returns'].mean()):
        optimized_sharpe_ratio = -np.inf
    else:
        optimized_sharpe_ratio = signals['net_returns'].mean() / signals['net_returns'].std() * np.sqrt(252)

    optimized_max_drawdown = (signals['cumulative_returns'].cummax() - signals['cumulative_returns']).max()

    print(f"Optimized Sharpe Ratio: {optimized_sharpe_ratio}")
    print(f"Optimized Max Drawdown: {optimized_max_drawdown}")

    return optimized_sharpe_ratio, optimized_max_drawdown



def run_kalman_filters(symbol, startDate, endDate):
    # Kalman filter
    # symbol = 'GOOG'  # 'AAPL'
    # startDate = '2020-05-22'
    # endDate = '2024-05-22'
    close_prices = download_close_prices(symbol, startDate, endDate)
    kalman_avg, spread, zscore = calc_kalman_filter(close_prices)
    # Define a range of thresholds to test
    long_entry_thresholds = np.arange(-5, -2, 1)
    short_entry_thresholds = np.arange(2, 5, 1)
    long_exit_thresholds = np.arange(-0.5, 0.5, 0.25)
    short_exit_thresholds = np.arange(-0.5, 0.5, 0.25)
    stop_loss_thresholds = np.arange(0.02, 0.05, 0.01)  # Stop-loss as a percentage of entry price
    best_sharpe_ratio, best_thresholds, results_df = get_best_thresholds(close_prices, kalman_avg, spread, zscore,
                                                                         long_entry_thresholds, short_entry_thresholds,
                                                                         long_exit_thresholds, short_exit_thresholds,
                                                                         stop_loss_thresholds)
    plot_cumulative_returns(symbol, close_prices, kalman_avg, spread, zscore, best_thresholds)


if __name__ == '__main__':
    run_kalman_filters('AMZN', '2020-05-22', '2024-05-22')


"""
Example using AAPL historical data:
Best Thresholds:
    Long Entry Threshold: -3
    Short Entry Threshold: 3
    Long Exit Threshold: 0.25
    Short Exit Threshold: -0.5
    Stop-Loss Threshold: 0.02
Performance Metrics:
    Best Sharpe Ratio: 0.6404972725290812
    Optimized Sharpe Ratio: 0.6404972725290812
    Optimized Max Drawdown: 0.1802083236506904
The strategy demonstrates a reasonable Sharpe Ratio of 0.6405, indicating a balance between risk and return.
The maximum drawdown of approximately 18% suggests that the strategy can withstand market downturns to a certain extent. 
"""