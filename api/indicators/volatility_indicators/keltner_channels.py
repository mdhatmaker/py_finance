import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def calculate_keltner_channel(data, ema_period=20, atr_period=10, multiplier=2):
    # Calculate the EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Calculate the ATR
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=atr_period).mean()

    # Calculate the upper and lower bands
    data['Upper'] = data['EMA'] + (data['ATR'] * multiplier)
    data['Lower'] = data['EMA'] - (data['ATR'] * multiplier)

    return data


def generate_buy_sell(data):
    # Buy when the close is higher than the upper band
    buy_signals = (data['Close'] < data['Lower'])
    # Sell when the close is lower than the lower band
    sell_signals = (data['Close'] > data['Upper'])
    return buy_signals, sell_signals


def plot_keltner_channels(ticker, data):
    # Plotting
    fig, ax = plt.subplots(figsize=(25, 8))

    # Stock price plot with Keltner Channels and buy/sell signals
    ax.plot(data['Close'], label='Close Price', alpha=0.5)
    ax.plot(data['EMA'], label='EMA (20 periods)', color='blue', alpha=0.6)
    ax.plot(data['Upper'], label='Upper Keltner Channel', linestyle='--', alpha=0.4)
    ax.plot(data['Lower'], label='Lower Keltner Channel', linestyle='--', alpha=0.4)
    ax.fill_between(data.index, data['Lower'], data['Upper'], color='grey', alpha=0.1)
    ax.scatter(data.index[data['Buy Signals']], data['Close'][data['Buy Signals']], label='Buy Signal', marker='^', color='green', alpha=1)
    ax.scatter(data.index[data['Sell Signals']], data['Close'][data['Sell Signals']], label='Sell Signal', marker='v', color='red', alpha=1)
    ax.set_title(f'{ticker} Stock Price with Keltner Channels')
    ax.set_ylabel('Price')
    ax.legend()

    plt.tight_layout()
    plt.show()


def run_keltner_channels(ticker: str, start_date: str, end_date: str):
    # Download data
    data = yf.download(ticker, start_date, end_date)
    # Calculate Keltner Channels
    data = calculate_keltner_channel(data)
    # Generate buy and sell signals
    buy_signals, sell_signals = generate_buy_sell(data)
    data['Buy Signals'] = buy_signals
    data['Sell Signals'] = sell_signals
    # Plot
    plot_keltner_channels(ticker, data)


if __name__ == "__main__":

    run_keltner_channels("AAPL", "2020-01-01", "2024-01-01")

