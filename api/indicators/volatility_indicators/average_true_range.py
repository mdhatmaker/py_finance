import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


# Function to calculate the Average True Range (ATR)
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def generate_buy_sell(data):
    # Calculate ATR and its moving average
    data['ATR'] = calculate_atr(data)
    data['ATR_MA'] = data['ATR'].rolling(window=14).mean()  # 14-day moving average of ATR

    # Define buy and sell signals
    buy_signal = (data['ATR'] > data['ATR_MA']) & (data['ATR'].shift(1) <= data['ATR_MA'].shift(1))
    sell_signal = (data['ATR'] < data['ATR_MA']) & (data['ATR'].shift(1) >= data['ATR_MA'].shift(1))

    return buy_signal, sell_signal


def plot_average_true_range(ticker, data, buy_signal, sell_signal):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 8), sharex=True)  # Share x-axis

    # Stock price plot with ATR-based buy and sell signals
    ax1.plot(data['Close'], label='Close Price', alpha=0.5)
    ax1.scatter(data.index[buy_signal], data['Close'][buy_signal], label='Buy Signal (ATR)', marker='^', color='green', alpha=1)
    ax1.scatter(data.index[sell_signal], data['Close'][sell_signal], label='Sell Signal (ATR)', marker='v', color='red', alpha=1)
    for idx in data.index[buy_signal]:
        ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
    for idx in data.index[sell_signal]:
        ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    ax1.set_title(f'{ticker} Stock Price with ATR-Based Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # ATR subplot with buy and sell signals
    ax2.plot(data['ATR'], label='Average True Range', color='purple')
    ax2.plot(data['ATR_MA'], label='14-day MA of ATR', color='orange', alpha=0.6)
    ax2.scatter(data.index[buy_signal], data['ATR'][buy_signal], label='Buy Signal (ATR)', marker='^', color='green')
    ax2.scatter(data.index[sell_signal], data['ATR'][sell_signal], label='Sell Signal (ATR)', marker='v', color='red')
    for idx in data.index[buy_signal]:
        ax2.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
    for idx in data.index[sell_signal]:
        ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    ax2.set_title(f'{ticker} Average True Range (ATR) with Signals')
    ax2.set_ylabel('ATR')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_average_true_range(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    buy_signal, sell_signal = generate_buy_sell(data)
    plot_average_true_range(ticker, data, buy_signal, sell_signal)


if __name__ == "__main__":

    # Sample indicator usage
    run_average_true_range("AAPL", "2020-01-01", "2023-01-01")

