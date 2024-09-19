import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def calculate_chaikin_volatility(data, ema_period=10, change_period=10):
    # High-Low spread
    data['HL_Spread'] = data['High'] - data['Low']

    # EMA of the High-Low spread
    data['EMA_HL_Spread'] = data['HL_Spread'].ewm(span=ema_period, adjust=False).mean()

    # Chaikin Volatility
    data['Chaikin_Volatility'] = (data['EMA_HL_Spread'] - data['EMA_HL_Spread'].shift(change_period)) / data['EMA_HL_Spread'].shift(
        change_period) * 100

    return data


def plot_volatility_chaikin(ticker, data):
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Stock price plot
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Chaikin Volatility plot
    ax2.plot(data['Chaikin_Volatility'], label='Chaikin Volatility', color='orange')
    ax2.set_title('Chaikin Volatility Indicator')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_volatility_chaikin(ticker: str, start_date: str, end_date: str):
    # Download data
    data = yf.download(ticker, start_date, end_date)
    # Calculate Chaikin Volatility
    data = calculate_chaikin_volatility(data)
    # Plot
    plot_volatility_chaikin(ticker, data)


if __name__ == "__main__":

    run_volatility_chaikin("AAPL", "2020-01-01", "2024-01-01")



