import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def calculate_donchian_channel(data, window=20):
    data['Upper'] = data['High'].rolling(window=window).max()
    data['Lower'] = data['Low'].rolling(window=window).min()
    data['Middle'] = (data['Upper'] + data['Lower']) / 2
    return data


def generate_buy_sell(data, window=60):
    # Calculate Donchian Channels with 20-day period
    data = calculate_donchian_channel(data, window)
    # Buy when the close price crosses above the upper band from below
    buy_signals = (data['Close'] < data['Lower'].shift(1)) & (data['Close'].shift(1) >= data['Lower'].shift(1))
    # Sell when the close price crosses below the lower band from above
    sell_signals = (data['Close'] > data['Upper'].shift(1)) & (data['Close'].shift(1) <= data['Upper'].shift(1))
    return buy_signals, sell_signals


def plot_donchian_channels(ticker, data):
    # Plotting
    fig, ax = plt.subplots(figsize=(25, 8))

    # Stock price plot with Donchian Channels and buy/sell signals
    ax.plot(data['Close'], label='Close Price', alpha=0.5)
    ax.plot(data['Upper'], label='Upper Donchian Channel', linestyle='--', alpha=0.4)
    ax.plot(data['Lower'], label='Lower Donchian Channel', linestyle='--', alpha=0.4)
    ax.plot(data['Middle'], label='Middle Donchian Channel', linestyle='--', alpha=0.4)
    ax.fill_between(data.index, data['Lower'], data['Upper'], color='grey', alpha=0.1)

    # Mark buy and sell signals
    ax.scatter(data.index[data['Buy Signals']], data['Close'][data['Buy Signals']], label='Buy Signal', marker='^', color='green', alpha=1)
    ax.scatter(data.index[data['Sell Signals']], data['Close'][data['Sell Signals']], label='Sell Signal', marker='v', color='red', alpha=1)

    ax.set_title(f'{ticker} Stock Price with Donchian Channels')
    ax.set_ylabel('Price')
    ax.legend()

    plt.tight_layout()
    plt.show()


def run_donchian_channels(ticker: str, start_date: str, end_date: str):
    # Download data
    data = yf.download(ticker, start_date, end_date)
    # Generate buy and sell signals
    buy_signals, sell_signals = generate_buy_sell(data)
    data['Buy Signals'] = buy_signals
    data['Sell Signals'] = sell_signals
    # Plot
    plot_donchian_channels(ticker, data)


if __name__ == "__main__":

    run_donchian_channels("AAPL", "2020-01-01", "2024-01-01")

