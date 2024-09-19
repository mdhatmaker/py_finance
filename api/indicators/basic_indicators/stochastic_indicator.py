import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api.utils.yahoo_finance import download_close_prices, download_ohlc


def calc_stochastic_oscillator(data, period=14):
    highs = data[:, 0]
    lows = data[:, 1]
    closes = data[:, 2]

    highs_max = np.maximum.accumulate(highs)
    lows_min = np.minimum.accumulate(lows)

    fast_k = 100 * ((closes - lows_min) / (highs_max - lows_min))
    slow_k = np.zeros_like(fast_k)

    for i in range(period, len(fast_k)):
        slow_k[i] = np.mean(fast_k[i - period:i])

    return fast_k, slow_k


def calculate_stochastic_oscillator(data, period=14):
    highs = data['High']
    lows = data['Low']
    closes = data['Close']

    highs_max = highs.rolling(window=period).max()
    lows_min = lows.rolling(window=period).min()

    fast_k = 100 * ((closes - lows_min) / (highs_max - lows_min))
    slow_k = fast_k.rolling(window=3).mean()

    return fast_k, slow_k


# def generate_random_data():
#     # Generating random data
#     np.random.seed(0)  # for reproducibility
#     data_points = 500
#     data = np.random.rand(data_points, 3) * 100  # High, Low, Close
#     return data


def plot_stochastic_indicator(ticker, data, fast_k, slow_k, period):
    # Plotting
    plt.figure(figsize=(14, 7))
    # plt.plot(data[:, 2], label='Close Price')
    plt.plot(data['Close'], label='Close Price')
    plt.plot(fast_k, label='%K', color='b')
    plt.plot(slow_k, label='%D', color='r')
    plt.title(f'{ticker} Stochastic Oscillator ({period}-period)')
    plt.xlabel('Data Points')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_stochastic_indicator(ticker, start_date, end_date, period=14):
    # data = generate_random_data()
    # prices = download_close_prices(ticker, start_date, end_date)
    # prices = np.asarray(prices)
    prices = download_ohlc(ticker, start_date, end_date)
    # Calculate stochastic oscillator
    # fast_k, slow_k = calc_stochastic_oscillator(prices, period=period)
    fast_k, slow_k = calculate_stochastic_oscillator(prices, period=period)
    plot_stochastic_indicator(ticker, prices, fast_k, slow_k, period)


# def stochastic_oscillator(data, period=14):
#     highs = data['High']
#     lows = data['Low']
#     closes = data['Close']
#
#     highs_max = highs.rolling(window=period).max()
#     lows_min = lows.rolling(window=period).min()
#
#     fast_k = 100 * ((closes - lows_min) / (highs_max - lows_min))
#     slow_k = fast_k.rolling(window=3).mean()
#
#     return fast_k, slow_k
#
#
# # Read data from CSV
# file_path = 'your_file_path.csv'  # Update with your CSV file path
# data = pd.read_csv(file_path)
# # Calculate stochastic oscillator
# fast_k, slow_k = stochastic_oscillator(data)


if __name__ == "__main__":
    run_stochastic_indicator('AAPL', '2020-01-01', None)

