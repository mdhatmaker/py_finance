import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from api.utils.yahoo_finance import download_close_prices, download_ohlc


# https://www.askpython.com/python/examples/relative-strength-index-rsi


def calculate_rsi(prices, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given list of prices.

    Args:
        prices: A list of closing prices.
        period: The number of periods for the moving average (default: 14).

    Returns:
        A pandas Series containing the RSI values for each price.
    """
    delta = prices.diff()
    delta = delta.dropna()  # Remove NaN from the first difference
    up, down = delta.clip(lower=0), delta.clip(upper=0, lower=None)  # Separate gains and losses

    ema_up = up.ewm(alpha=1 / period, min_periods=period).mean()  # Exponential Moving Average for gains
    ema_down = down.abs().ewm(alpha=1 / period, min_periods=period).mean()  # EMA for absolute losses

    rs = ema_up / ema_down  # Average gain / Average loss
    rsi = 100 - 100 / (1 + rs)  # Calculate RSI

    return rsi


# def generate_random_prices(num_days=1000):
#     """
#     Generates a list of random stock prices for a specified number of days.
#
#     Args:
#         num_days: The number of days to generate prices for (default: 1000).
#
#     Returns:
#         A list of random closing prices.
#     """
#     # Simulate price fluctuations with a random walk
#     prices = [100]  # Starting price
#     for _ in range(num_days - 1):
#         change = random.uniform(-1, 1) * 0.5  # Random change between -0.5 and 0.5
#         prices.append(prices[-1] + change)
#
#     return prices


def plot_rsi_indicator(ticker, df, period):
    # Calculate RSI using pandas for efficiency
    # df = pd.DataFrame({'Close': prices})
    df['RSI'] = calculate_rsi(df['Close'], period=period)
    # Plot the closing prices and RSI
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.xlabel('Day')
    plt.ylabel('Price/RSI')
    plt.title(f'{ticker} Stock Price with RSI ({period}-period)')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_rsi_indicator(ticker, start_date, end_date, period=14):
    # prices = generate_random_prices()
    # prices = download_close_prices(ticker, start_date, end_date)
    # prices = np.asarray(prices)
    prices = download_ohlc(ticker, start_date, end_date)
    plot_rsi_indicator(ticker, prices, period=period)


if __name__ == "__main__":
    run_rsi_indicator('AAPL', '2023-01-01', None)

