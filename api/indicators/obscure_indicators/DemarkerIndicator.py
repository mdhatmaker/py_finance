import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def calculate_demarker(data, period=14):
    data['DeMax'] = 0
    data['DeMin'] = 0

    for i in range(1, len(data)):
        if data['Close'][i] > data['Close'][i - 1]:
            data['DeMax'][i] = data['Close'][i] - data['Close'][i - 1]
        elif data['Close'][i] < data['Close'][i - 1]:
            data['DeMin'][i] = data['Close'][i - 1] - data['Close'][i]

    data['DeMM'] = data['DeMax'].rolling(window=period).mean()
    data['DeMn'] = data['DeMin'].rolling(window=period).mean()

    data['DeM'] = data['DeMM'] / (data['DeMM'] + data['DeMn'])
    return data


def plot_stock_with_demarker(stock_symbol, start_date, end_date, period=14):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = calculate_demarker(data, period)

    # Generating Buy/Sell signals
    data['Buy_Signal'] = (data['DeM'] < 0.3) & (data['DeM'].shift(1) >= 0.3)
    data['Sell_Signal'] = (data['DeM'] > 0.7) & (data['DeM'].shift(1) <= 0.7)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # Stock price with buy and sell signals
    ax1.plot(data['Close'], label='Close Price', alpha=0.5)
    ax1.scatter(data.index[data['Buy_Signal']], data['Close'][data['Buy_Signal']], label='Buy Signal', marker='^', color='green', s=100)
    ax1.scatter(data.index[data['Sell_Signal']], data['Close'][data['Sell_Signal']], label='Sell Signal', marker='v', color='red', s=100)
    ax1.set_title(f'{stock_symbol} Stock Price with Buy and Sell Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # DeMarker Indicator
    ax2.plot(data['DeM'], label='DeMarker', color='blue')
    ax2.axhline(0.7, color='red', linestyle='--', label='Overbought Threshold (0.7)')
    ax2.axhline(0.3, color='green', linestyle='--', label='Oversold Threshold (0.3)')
    ax2.set_title(f'{stock_symbol} DeMarker Indicator')
    ax2.set_ylabel('DeMarker Value')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_demarker_indicator(ticker: str, start_date: str, end_date: str):
    # plot_stock_with_demarker('JNJ', '2022-01-01', '2024-01-01')
    plot_stock_with_demarker(ticker, start_date, end_date)

