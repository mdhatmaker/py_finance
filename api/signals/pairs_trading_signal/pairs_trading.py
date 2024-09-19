import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
from itertools import combinations
from statsmodels.tsa.stattools import coint
from pandas import Series

from api.signals.pairs_trading_signal.PairsTradingStrategy import PairsTradingStrategy


# https://medium.com/@thepythonlab/implementing-a-pairs-trading-strategy-in-python-a-step-by-step-guide-f0c38bb320cd


def download_asset_prices(assets: List[str], start_date: str = '2020-05-22', end_date: str = '2024-05-22') -> Series:
    # Download historical stock data from Yahoo Finance
    print(f'{assets}    {start_date} to {end_date}')
    # assets = ['AAPL', 'MSFT']
    # start_date = '2019-01-01'
    # end_date = '2023-04-30'
    data = yf.download(assets, start=start_date, end=end_date)

    # Calculate daily returns of the assets
    returns = data['Adj Close'].pct_change().dropna()
    return returns


def plot_returns(assets, returns):
    # Plot the daily returns
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(returns.index, returns[asset], label=asset)

    plt.title('Daily Returns of Selected Assets')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_correlation_heatmap(correlation_matrix):
    # Plot the correlation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Heatmap of Selected Assets')
    plt.show()
    plt.show()


def find_correlated_pairs(assets, returns):
    # Calculate the correlation matrix of asset returns
    correlation_matrix = returns.corr()

    # Find pairs with high positive correlation
    highly_correlated_pairs = []
    for pair in combinations(assets, 2):
        asset1, asset2 = pair
        correlation = correlation_matrix.loc[asset1, asset2]
        if correlation > 0.7:
            highly_correlated_pairs.append(pair)

    # Print highly correlated pairs
    print("Highly Correlated Pairs:")
    for pair in highly_correlated_pairs:
        print(pair)

    return correlation_matrix, highly_correlated_pairs


# Implement cointegration analysis for pair selection
def cointegration_analysis(assets, data):
    cointegrated_pairs = []

    for pair in combinations(assets, 2):
        asset1, asset2 = pair
        result = coint(data[asset1], data[asset2])

        if result[1] < 0.05:  # Check for p-value significance
            cointegrated_pairs.append(pair)

    # Print cointegrated pairs
    print("Cointegrated Pairs:")
    for pair in cointegrated_pairs:
        print(pair)

    return cointegrated_pairs


def plot_cointegrated_pairs(cointegrated_pairs, returns):
    # Plot the cointegrated pairs
    plt.figure(figsize=(12, 6))
    for pair in cointegrated_pairs:
        asset1, asset2 = pair
        plt.plot(returns.index, returns[asset1]-returns[asset2], label=f"{asset1}-{asset2}")

    plt.title('Spread between Cointegrated Pairs')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_pairs_trading_signal(assets: List[str], start_date: str, end_date: str=None):
    # assets = ['AAPL', 'MSFT']
    # start_date = '2019-01-01'
    # end_date = '2023-04-30'
    if not end_date:
        end_date = datetime.now()

    returns = download_asset_prices(assets, start_date, end_date)
    plot_returns(assets, returns)

    correlation_matrix, highly_correlated_pairs = find_correlated_pairs(assets, returns)
    plot_correlation_heatmap(correlation_matrix)

    # Perform cointegration analysis on the returns data
    cointegrated_pairs = cointegration_analysis(assets, returns)
    plot_cointegrated_pairs(cointegrated_pairs, returns)

    # Create PairsTradingStrategy object for selected assets
    strategy = PairsTradingStrategy(returns, 'AAPL', 'MSFT')
    strategy.fit()
    # Generate trading signals based on z-score threshold
    signals = strategy.generate_signals(zscore_threshold=1)
    # Execute trades based on signals
    strategy.execute_trades(signals)


