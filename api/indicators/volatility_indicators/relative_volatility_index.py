import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def calculate_rvi(data, period=14):
    # Calculate the standard deviation of high and low for a given period
    data['StdDev_High'] = data['High'].rolling(window=period).std()
    data['StdDev_Low'] = data['Low'].rolling(window=period).std()

    # Calculate the RVI
    data['RVI'] = 100 * (data['StdDev_High'] / (data['StdDev_High'] + data['StdDev_Low']))

    # Calculate the Signal line, which is a 4-period SMA of the RVI
    data['RVI_Signal'] = data['RVI'].rolling(window=4).mean()

    return data


def generate_buy_sell(data, signal_threshold=5):
    # Buy when RVI crosses above the signal line with a minimum threshold
    data['Buy Signals'] = ((data['RVI'] > data['RVI_Signal']) &
                           (data['RVI'] - data['RVI_Signal'] > signal_threshold) &
                           (data['RVI'].shift(1) <= data['RVI_Signal'].shift(1)))

    # Sell when RVI crosses below the signal line with a minimum threshold
    data['Sell Signals'] = ((data['RVI'] < data['RVI_Signal']) &
                            (data['RVI_Signal'] - data['RVI'] > signal_threshold) &
                            (data['RVI'].shift(1) >= data['RVI_Signal'].shift(1)))
    return data


def plot_relative_volatility_index(ticker, data):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Stock price plot with buy and sell signals
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.scatter(data.index[data['Buy Signals']], data['Close'][data['Buy Signals']], label='Buy Signal', marker='^', color='green', alpha=1)
    ax1.scatter(data.index[data['Sell Signals']], data['Close'][data['Sell Signals']], label='Sell Signal', marker='v', color='red', alpha=1)

    '''
    for idx in data.index[data['Buy Signals']]:
        ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=idx, color='green', linestyle='--', alpha=0.5)
    
    for idx in data.index[data['Sell Signals']]:
        ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.5)
    '''

    ax1.set_title(f'{ticker} Stock Price with RVI Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # RVI plot with buy and sell signals
    ax2.plot(data['RVI'], label='RVI', color='green')
    ax2.plot(data['RVI_Signal'], label='Signal Line', color='red', linestyle='--')
    ax2.scatter(data.index[data['Buy Signals']], data['RVI'][data['Buy Signals']], label='Buy Signal', marker='^', color='blue', alpha=1)
    ax2.scatter(data.index[data['Sell Signals']], data['RVI'][data['Sell Signals']], label='Sell Signal', marker='v', color='orange', alpha=1)
    ax2.set_title('Relative Volatility Index (RVI) with Signals')
    ax2.set_ylabel('RVI')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_relative_volatility_index(ticker: str, start_date: str, end_date: str):
    # Download data
    data = yf.download(ticker, start_date, end_date)
    # Calculate RVI
    data = calculate_rvi(data)
    # Generate buy and sell signals with a defined threshold
    signal_threshold = 1  # Adjust this threshold to control signal strength
    data = generate_buy_sell(data, signal_threshold)
    # Plot
    plot_relative_volatility_index(ticker, data)


if __name__ == "__main__":

    run_relative_volatility_index("AAPL", "2020-01-01", "2024-01-01")

