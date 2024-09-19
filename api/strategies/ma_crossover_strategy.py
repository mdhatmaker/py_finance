import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from api.utils.yahoo_finance import download_ohlc


# https://www.askpython.com/python/examples/moving-average-crossover-strategy
# https://www.askpython.com/resources/maximizing-cost-savings-through-offshore-development


# Define number of data points and window sizes
# data_points = 500
# short_window = 20
# long_window = 50
# # Generate random closing prices with some volatility
# np.random.seed(10)  # Set seed for reproducibility
# closing_prices = np.random.normal(100, 5, data_points) + np.random.rand(data_points) * 10
# # Create pandas DataFrame
# data = pd.DataFrame(closing_prices, columns=['Close'])


def calc_ma_crossover_strategy(data, short_window=20, long_window=50):
    # Calculate Simple Moving Averages (SMA)
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

    # Calculate Exponential Moving Averages (EMA) with smoothing factor (alpha)
    alpha = 2 / (short_window + 1)
    data['EMA_Short'] = data['Close'].ewm(alpha=alpha, min_periods=short_window).mean()

    alpha = 2 / (long_window + 1)
    data['EMA_Long'] = data['Close'].ewm(alpha=alpha, min_periods=long_window).mean()

    # Generate Buy/Sell Signals based on crossovers
    data['Signal_SMA'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, 0)
    data['Position_SMA'] = data['Signal_SMA'].diff()

    data['Signal_EMA'] = np.where(data['EMA_Short'] > data['EMA_Long'], 1, 0)
    data['Position_EMA'] = data['Signal_EMA'].diff()

    return data


def plot_ma_crossover_strategy(ticker, data, short_window, long_window):
    # Plot closing prices, SMAs, and EMAs
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Closing Price')
    plt.plot(data.index, data['SMA_Short'], label=f'SMA ({short_window})')
    plt.plot(data.index, data['SMA_Long'], label=f'SMA ({long_window})')
    plt.plot(data.index, data['EMA_Short'], label=f'EMA ({short_window})')
    plt.plot(data.index, data['EMA_Long'], label=f'EMA ({long_window})')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Prices with Moving Averages')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Print buy/sell signals for SMA and EMA
    print("Buy/Sell Signals (SMA):")
    print(data[['Close', 'Signal_SMA', 'Position_SMA']].tail(20))
    print("\nBuy/Sell Signals (EMA):")
    print(data[['Close', 'Signal_EMA', 'Position_EMA']].tail(20))


# def plot_moving_average(filepath, short_window, long_window):
#     """
#     Plot the moving average of a CSV or Excel file.
#
#     Args:
#         filepath (str): The path to the CSV or Excel file.
#         short_window (int): The window size for the short moving average.
#         long_window (int): The window size for the long moving average.
#     """
#     data = pd.read_csv(filepath)
#     data.set_index("Date", inplace=True)
#
#     data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
#     data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()
#
#     data["Signal_SMA"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, 0)
#     data["Position_SMA"] = data["Signal_SMA"].diff()
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(data.index, data["Close"], label="Closing Price")
#     plt.plot(data.index, data["SMA_Short"], label=f"SMA ({short_window})")
#     plt.plot(data.index, data["SMA_Long"], label=f"SMA ({long_window})")
#     plt.xlabel("Date")
#     plt.ylabel("Price")
#     plt.title("Moving Average Crossover")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def run_ma_crossover_strategy(ticker, start_date, end_date, short_window=20, long_window=50):
    data = download_ohlc(ticker, start_date, end_date)
    data = calc_ma_crossover_strategy(data, short_window, long_window)
    plot_ma_crossover_strategy(ticker, data, short_window, long_window)


if __name__ == "__main__":
    run_ma_crossover_strategy('AAPL', '2020-01-01', None, short_window=20, long_window=50)

