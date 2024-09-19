import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from api.utils.yahoo_finance import download_close_prices


# https://www.askpython.com/python/examples/bollinger-bands-python


def generate_random_prices():
    # Generate random stock prices
    np.random.seed(42)  # for reproducibility
    stock_prices = np.random.normal(100, 5, 250)
    return stock_prices


def calc_bollinger_bands(stock_prices, window_size = 20, num_std = 2):
    # # Define parameters
    # window_size = 20
    # num_std = 2

    # Calculate rolling mean and standard deviation
    rolling_mean = np.convolve(stock_prices, np.ones(window_size) / window_size, mode='valid')
    rolling_std = np.std([stock_prices[i:i + window_size] for i in range(len(stock_prices) - window_size + 1)], axis=1)

    # Calculate Bollinger Bands
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    return rolling_mean, upper_band, lower_band


def plot_bollinger_bands_indicator(ticker, stock_prices, rolling_mean, upper_band, lower_band, window_size = 20):
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(stock_prices, label='Stock Price')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(upper_band, label='Upper Bollinger Band', color='green')
    plt.plot(lower_band, label='Lower Bollinger Band', color='green')
    plt.fill_between(np.arange(window_size - 1, len(stock_prices)), lower_band, upper_band, color='grey', alpha=0.2)
    plt.title(f'{ticker} Bollinger Bands')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_bollinger_bands_indicator(ticker, start_date, end_date, window_size=20, num_std=2):
    # window_size = 20
    # num_std = 2
    # stock_prices = generate_random_prices()
    stock_prices = download_close_prices(ticker, start_date, end_date)
    stock_prices = np.asarray(stock_prices)

    rolling_mean, upper_band, lower_band = calc_bollinger_bands(stock_prices, window_size, num_std)
    plot_bollinger_bands_indicator(ticker, stock_prices, rolling_mean, upper_band, lower_band, window_size)



# # Read stock prices from Excel or CSV file
# # Replace 'stock_prices.xlsx' or 'stock_prices.csv' with your file name
# file_path = 'stock_prices.xlsx'
# # If your file is in CSV format, uncomment the following line and comment out the previous line
# # file_path = 'stock_prices.csv'
#
# # Assuming the file has a single column of stock prices named 'Price'
# stock_prices_df = pd.read_excel(file_path)  # for Excel
# # If your file is in CSV format, uncomment the following line and comment out the previous line
# # stock_prices_df = pd.read_csv(file_path)
#
# # Extract stock prices from DataFrame
# stock_prices = stock_prices_df['Price'].values
#
# # Define parameters
# window_size = 20
# num_std = 2
#
# # Calculate rolling mean and standard deviation
# rolling_mean = np.convolve(stock_prices, np.ones(window_size) / window_size, mode='valid')
# rolling_std = np.std([stock_prices[i:i + window_size] for i in range(len(stock_prices) - window_size + 1)], axis=1)
#
# # Calculate Bollinger Bands
# upper_band = rolling_mean + num_std * rolling_std
# lower_band = rolling_mean - num_std * rolling_std
#
# # Plotting
# plt.figure(figsize=(14, 7))
# plt.plot(stock_prices, label='Stock Price')
# plt.plot(rolling_mean, label='Rolling Mean', color='red')
# plt.plot(upper_band, label='Upper Bollinger Band', color='green')
# plt.plot(lower_band, label='Lower Bollinger Band', color='green')
# plt.fill_between(np.arange(window_size - 1, len(stock_prices)), lower_band, upper_band, color='grey', alpha=0.2)
# plt.title('Bollinger Bands')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()


if __name__ == "__main__":
    run_bollinger_bands_indicator('AAPL', '2023-01-01', None)

