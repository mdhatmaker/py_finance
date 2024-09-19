import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api.utils.yahoo_finance import download_close_prices


# def generate_stock_prices(n=500, start_price=100, volatility=0.05):
#     prices = [start_price]
#     for _ in range(1, n):
#         price = prices[-1] * (1 + np.random.normal(0, volatility))
#         prices.append(price)
#     return prices


def calculate_macd_indicator(prices, macd_values, short_window=12, long_window=26, signal_window=9):
    short_ema = np.mean(prices[-short_window:])
    long_ema = np.mean(prices[-long_window:])
    macd_line = short_ema - long_ema
    signal_line = np.mean(macd_values[-(signal_window + 1):-1])
    return macd_line, signal_line


def calc_macd_indicator(stock_prices):
    # Calculate MACD
    macd_values = []
    signal_values = []
    for i in range(26, len(stock_prices)):
        macd, signal = calculate_macd_indicator(stock_prices[:i + 1], macd_values)
        macd_values.append(macd)
        signal_values.append(signal)
    return signal_values, macd_values


def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, min_periods=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, min_periods=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=signal_window, adjust=False).mean()
    return signal_line, macd_line


def plot_macd_indicator(ticker, stock_prices, signal_values, macd_values):
    # Plot stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(stock_prices, label='Stock Prices', color='blue')
    # Plot MACD and Signal lines
    plt.plot(range(26, len(stock_prices)), macd_values, label='MACD Line', color='red')
    plt.plot(range(26, len(stock_prices)), signal_values, label='Signal Line', color='green')

    plt.title(f'{ticker} Stock Prices and MACD Indicator')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_macd(ticker, prices, signal_line, macd_line):
    # Plot stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Stock Prices', color='blue')
    # Plot MACD and Signal lines
    plt.plot(macd_line, label='MACD Line', color='red')
    plt.plot(signal_line, label='Signal Line', color='green')

    plt.title(f'{ticker} Stock Prices and MACD Indicator')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_macd_indicator(ticker, start_date, end_date):
    # Generate random stock prices
    # stock_prices = generate_stock_prices()
    stock_prices = download_close_prices(ticker, start_date, end_date)
    # stock_prices = np.asarray(stock_prices)
    # signal_values, macd_values = calc_macd_indicator(stock_prices)
    # plot_macd_indicator(ticker, stock_prices, signal_values, macd_values)
    signal_line, macd_line = calculate_macd(stock_prices)
    plot_macd(ticker, stock_prices, signal_line, macd_line)


# # Read stock prices from CSV
# data = pd.read_csv('stock_prices.csv')  # Replace 'stock_prices.csv' with your file name
# # Assuming the CSV file has a column named 'Close' containing the closing prices
# prices = data['Close']
# # Calculate MACD
# macd_line, signal_line = calculate_macd(prices)


if __name__ == "__main__":
    run_macd_indicator('AAPL', '2023-01-01', None)

