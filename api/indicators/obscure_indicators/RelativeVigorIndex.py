from typing import Optional
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from api.utils.yahoo_finance import download_ohlc


def calculate_rvi(data, period=10):
    numerator = data['Close'] - data['Open']
    denominator = data['High'] - data['Low']
    data['RVI'] = (numerator.rolling(window=period).mean() /
                   denominator.rolling(window=period).mean())
    data['RVI_Signal'] = data['RVI'].rolling(window=period).mean()
    return data


def generate_signals(data):
    data['Buy Signals'] = (data['RVI'] > data['RVI_Signal']) & (data['RVI'].shift(1) <= data['RVI_Signal'].shift(1))
    data['Sell Signals'] = (data['RVI'] < data['RVI_Signal']) & (data['RVI'].shift(1) >= data['RVI_Signal'].shift(1))
    return data


def plot_relative_vigor(ticker, data):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Stock price plot with buy and sell signals
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.scatter(data.index[data['Buy Signals']], data['Close'][data['Buy Signals']], label='Buy Signal', marker='^', color='green', alpha=1)
    ax1.scatter(data.index[data['Sell Signals']], data['Close'][data['Sell Signals']], label='Sell Signal', marker='v', color='red', alpha=1)
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    # RVI plot with buy and sell signals
    ax2.plot(data['RVI'], label='RVI', color='green')
    ax2.plot(data['RVI_Signal'], label='Signal Line', color='red', linestyle='--')
    ax2.scatter(data.index[data['Buy Signals']], data['RVI'][data['Buy Signals']], label='Buy Signal', marker='^', color='blue', alpha=1)
    ax2.scatter(data.index[data['Sell Signals']], data['RVI'][data['Sell Signals']], label='Sell Signal', marker='v', color='orange', alpha=1)
    ax2.set_title('Relative Vigor Index (RVI) with Signals')
    ax2.set_ylabel('RVI')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_relative_vigor(ticker: str, start_date: str, end_date: Optional[str]):
    data = download_ohlc(ticker, start_date, end_date)
    # Calculate RVI and generate signals
    data = calculate_rvi(data)
    data = generate_signals(data)
    plot_relative_vigor(ticker, data)


if __name__ == "__main__":

    run_relative_vigor('AAPL', '2020=01=01', None)

