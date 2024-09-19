import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def calculate_bollinger_bands(data, window=20, num_of_std=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band

def generate_buy_sell(data):
    # Calculate Bollinger Bands
    data['Middle Band'], data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)

    # Define buy and sell signals
    buy_signal = (data['Close'] < data['Lower Band']) & (data['Close'].shift(1) >= data['Lower Band'].shift(1))
    sell_signal = (data['Close'] > data['Upper Band']) & (data['Close'].shift(1) <= data['Upper Band'].shift(1))

    return buy_signal, sell_signal


def plot_bollinger_bands(ticker, data, buy_signal, sell_signal):
    # Plotting
    fig, ax = plt.subplots(figsize=(25, 8))

    # Stock price plot with Bollinger Bands and buy/sell signals
    ax.plot(data['Close'], label='Close Price', alpha=1, linewidth=2)
    ax.plot(data['Middle Band'], label='Middle Band (20-day SMA)', color='blue', alpha=0.5)
    ax.plot(data['Upper Band'], label='Upper Band (2 Std Dev)', color='red', alpha=0.5)
    ax.plot(data['Lower Band'], label='Lower Band (2 Std Dev)', color='green', alpha=0.5)
    ax.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='grey', alpha=0.1)
    ax.scatter(data.index[buy_signal], data['Close'][buy_signal], label='Buy Signal', marker='^', color='green', alpha=1)
    ax.scatter(data.index[sell_signal], data['Close'][sell_signal], label='Sell Signal', marker='v', color='red', alpha=1)
    ax.set_title(f'{ticker} Stock Price with Bollinger Bands')
    ax.set_ylabel('Price')
    ax.legend()

    plt.tight_layout()
    plt.show()


def run_bollinger_bands(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start_date, end_date)
    buy_signal, sell_signal = generate_buy_sell(data)
    plot_bollinger_bands(ticker, data, buy_signal, sell_signal)


if __name__ == "__main__":

    # Sample indicator usage
    run_bollinger_bands("AAPL", "2020-01-01", "2023-01-01")

