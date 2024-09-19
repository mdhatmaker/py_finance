import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from api.utils.yahoo_finance import download_ohlc


def aroon_oscillator(data, period=25):
    aroon_up = 100 * (data['High'].rolling(period + 1).apply(np.argmax, raw=True) / period)
    aroon_down = 100 * (data['Low'].rolling(period + 1).apply(np.argmin, raw=True) / period)
    aroon_osc = aroon_up - aroon_down
    return aroon_osc


def plot_aroon(data, aroon_osc, buy_signals, sell_signals):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 8), sharex=True)

    ax1.plot(data.index, data['Close'], label='ASML.AS Close Price', color='blue')
    ax1.plot(data.index[buy_signals], data.loc[buy_signals]['Close'], '^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(data.index[sell_signals], data.loc[sell_signals]['Close'], 'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_ylabel("Close Price")
    ax1.legend()

    ax2.plot(data.index, aroon_osc, label='Aroon Oscillator', color='purple')
    ax2.axhline(0, linestyle='--', color='black')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Aroon Oscillator")
    ax2.legend()

    for signal_date in data.index[buy_signals]:
        ax1.axvline(signal_date, color='g', linestyle='--', alpha=0.5)
        ax2.axvline(signal_date, color='g', linestyle='--', alpha=0.5)

    for signal_date in data.index[sell_signals]:
        ax1.axvline(signal_date, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(signal_date, color='r', linestyle='--', alpha=0.5)

    plt.show()


def run_aroon_oscillator(ticker: str, start_date: str, end_date: str):
    #data = yf.download(ticker, start=start_date, end=end_date)
    data = download_ohlc(ticker, start_date, end_date)

    # Calculate the Aroon Oscillator
    aroon_osc = aroon_oscillator(data)

    # Generate buy/sell signals
    buy_signals = (aroon_osc > 0) & (aroon_osc.shift(1) <= 0)
    sell_signals = (aroon_osc < 0) & (aroon_osc.shift(1) >= 0)

    # Plot the Aroon Oscillator and buy/sell signals
    plot_aroon(data, aroon_osc, buy_signals, sell_signals)


if __name__ == "__main__":

    # Fetch stock data from Yahoo Finance
    ticker = "AFX.DE"
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    run_aroon_oscillator(ticker, start_date, end_date)
