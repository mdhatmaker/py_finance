from itertools import combinations
from statsmodels.tsa.stattools import coint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
import numpy as np
import os


def double_simple_moving_average_signals(ticker_ts_df, short_window=5, long_window=30):
    """
    Generate trading signals based on a double simple moving average (SMA) strategy.
    Parameters:
    - aapl_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
    - short_window (int): The window size for the short-term SMA.
    - long_window (int): The window size for the long-term SMA.
    Returns:
    - signals (pandas.DataFrame): A DataFrame containing the trading signals.
    """
    signals = pd.DataFrame(index=ticker_ts_df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = ticker_ts_df['Close'].rolling(window=short_window,
                                                          min_periods=1,
                                                          center=False).mean()
    signals['long_mavg'] = ticker_ts_df['Close'].rolling(window=long_window,
                                                         min_periods=1,
                                                         center=False).mean()
    # Generate signal when SMAs cross
    signals['signal'] = np.where(
        signals['short_mavg'] > signals['long_mavg'], 1, 0)
    signals['orders'] = signals['signal'].diff()
    signals.loc[signals['orders'] == 0, 'orders'] = None
    return signals

def load_ticker_ts_df(ticker, start_date, end_date):
    """
    Load and cache time series financial data from Yahoo Finance API.
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    - start_date (str): The start date in 'YYYY-MM-DD' format for data retrieval.
    - end_date (str): The end date in 'YYYY-MM-DD' format for data retrieval.
    Returns:
    - df (pandas.DataFrame): A DataFrame containing the financial time series data.

    """
    dir_path = '/Users/michael/git/nashed/t1/webTraderApp/files'
    cached_file_path = f'{dir_path}/data/{ticker}_{start_date}_{end_date}.pkl'
    try:
        if os.path.exists(cached_file_path):
            df = pd.read_pickle(cached_file_path)
        else:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            df.to_pickle(cached_file_path)
    except FileNotFoundError:
        print(
            f'Error downloading and caching or loading file with ticker: {ticker}')

    return df


def calculate_profit(signals, prices):
    """
    Calculate cumulative profit based on trading signals and stock prices.
    Parameters:
    - signals (pandas.DataFrame): A DataFrame containing trading signals (1 for buy, -1 for sell).
    - prices (pandas.Series): A Series containing stock prices corresponding to the signal dates.
    Returns:
    - cum_profit (pandas.Series): A Series containing cumulative profit over time.
    """
    profit = pd.DataFrame(index=prices.index)
    profit['profit'] = 0.0

    buys = signals[signals['orders'] == 1].index
    sells = signals[signals['orders'] == -1].index
    while sells[0] < buys[0]:
        # These are long only strategies, we cannot start with sell
        sells = sells[1:]

    if len(buys) == 0 or len(sells) == 0:
        # no actions taken
        return profit
    if len(sells) < len(buys):
        # Assume we sell at the end
        sells = sells.append(pd.Index(prices.tail(1).index))

    buy_prices = prices.loc[buys]
    sell_prices = prices.loc[sells]

    profit.loc[sells, 'profit'] = sell_prices.values - buy_prices.values
    profit['profit'] = profit['profit'].fillna(0)

    # Make profit cumulative
    profit['cum_profit'] = profit['profit'].cumsum()

    return profit['cum_profit']


def plot_strategy(prices_df, signal_df, profit):
    """
    Plot a trading strategy with buy and sell signals and cumulative profit.
    Parameters:
    - prices (pandas.Series): A Series containing stock prices.
    - signals (pandas.DataFrame): A DataFrame with buy (1) and sell (-1) signals.
    - profit (pandas.Series): A Series containing cumulative profit over time.
    Returns:
    - ax1 (matplotlib.axes.Axes): The top subplot displaying stock prices and signals.
    - ax2 (matplotlib.axes.Axes): The bottom subplot displaying cumulative profit.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': (3, 1)},
                                   figsize=(18, 12))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price in $')
    ax1.plot(prices_df.index, prices_df, color='g', lw=0.25)

    # Plot the Buy and Sell signals
    ax1.plot(signal_df.loc[signal_df.orders == 1.0].index,
             prices_df[signal_df.orders == 1.0],
             '^', markersize=12, color='blue', label='Buy')
    ax1.plot(signal_df.loc[signal_df.orders == -1.0].index,
             prices_df[signal_df.orders == -1.0],
             'v', markersize=12, color='red', label='Sell')

    ax2.plot(profit.index, profit, color='b')
    ax2.set_ylabel('Cumulative Profit (%)')
    ax2.set_xlabel('Date')

    return ax1, ax2


def naive_momentum_signals(ticker_ts_df, nb_conseq_days=2):
    """
    Generate naive momentum trading signals based on consecutive positive or negative price changes.
    Parameters:
    - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
    - nb_conseq_days (int): The number of consecutive positive or negative days to trigger a signal.
    Returns:
    - signals (pandas.DataFrame): A DataFrame with 'orders' column containing buy (1) and sell (-1) signals.
    """
    signals = pd.DataFrame(index=ticker_ts_df.index)
    signals['orders'] = 0

    price = ticker_ts_df['Adj Close']
    price_diff = price.diff()

    signal = 0
    cons_day = 0

    for i in range(1, len(ticker_ts_df)):
        if price_diff[i] > 0:
            cons_day = cons_day + 1 if price_diff[i] > 0 else 0
            if cons_day == nb_conseq_days and signal != 1:
                signals['orders'].iloc[i] = 1
                signal = 1
        elif price_diff[i] < 0:
            cons_day = cons_day - 1 if price_diff[i] < 0 else 0
            if cons_day == -nb_conseq_days and signal != -1:
                signals['orders'].iloc[i] = -1
                signal = -1

    return signals


def mean_reversion_signals(ticker_ts_df, entry_threshold=1.0, exit_threshold=0.5):
    """
    Generate mean reversion trading signals based on moving averages and thresholds.
    Parameters:
    - ticker_ts_df (pandas.DataFrame): A DataFrame containing historical stock data.
    - entry_threshold (float): The entry threshold as a multiple of the standard deviation.
    - exit_threshold (float): The exit threshold as a multiple of the standard deviation.

    Returns:
    - signals (pandas.DataFrame): A DataFrame with 'orders' column containing buy (1) and sell (-1) signals.

    """
    signals = pd.DataFrame(index=ticker_ts_df.index)
    signals['mean'] = ticker_ts_df['Adj Close'].rolling(
        window=20).mean()  # Adjust the window size as needed
    signals['std'] = ticker_ts_df['Adj Close'].rolling(
        window=20).std()  # Adjust the window size as needed

    signals['signal'] = np.where(ticker_ts_df['Adj Close'] > (
        signals['mean'] + entry_threshold * signals['std']), 1, 0)
    signals['signal'] = np.where(ticker_ts_df['Adj Close'] < (
        signals['mean'] - exit_threshold * signals['std']), -1, 0)

    signals['orders'] = signals['signal'].diff()
    signals.loc[signals['orders'] == 0, 'orders'] = None

    return signals



def run_naive_momentum(ticker: str, start_date: str, end_date: str):
    # ts_df = load_ticker_ts_df('AAPL', start_date='2021-01-01', end_date='2023-01-01')
    ts_df = load_ticker_ts_df(ticker, start_date, end_date)

    signal_df = naive_momentum_signals(ts_df)
    profit_series = calculate_profit(signal_df, ts_df["Adj Close"])

    ax1, _ = plot_strategy(ts_df["Adj Close"], signal_df, profit_series)

    ax1.legend(loc='upper left', fontsize=10)
    plt.show()


def run_sma_crossover(ticker: str, start_date: str, end_date: str):
    ts_df = load_ticker_ts_df(ticker, start_date, end_date)

    signal_df = double_simple_moving_average_signals(ts_df, 5, 30)
    profit_series = calculate_profit(signal_df, ts_df["Adj Close"])

    ax1, ax2 = plot_strategy(ts_df["Adj Close"], signal_df, profit_series)

    # Add short and long moving averages
    ax1.plot(signal_df.index, signal_df['short_mavg'],
             linestyle='--', label='Fast SMA')
    ax1.plot(signal_df.index, signal_df['long_mavg'],
             linestyle='--', label='Slow SMA')
    ax1.legend(loc='upper left', fontsize=10)
    plt.show()

def run_mean_reversion(ticker: str, start_date: str, end_date: str):
    ts_df = load_ticker_ts_df(ticker, start_date, end_date)

    signal_df = mean_reversion_signals(ts_df)
    profit_series = calculate_profit(signal_df, ts_df["Adj Close"])

    ax1, _ = plot_strategy(ts_df["Adj Close"], signal_df, profit_series)

    ax1.plot(signal_df.index, signal_df['mean'], linestyle='--', label="Mean")
    ax1.plot(signal_df.index, signal_df['mean'] +
             signal_df['std'], linestyle='--', label="Ceiling STD")
    ax1.plot(signal_df.index, signal_df['mean'] -
             signal_df['std'], linestyle='--', label="Floor STD")
    ax1.legend(loc='upper left', fontsize=10)
    plt.show()


if __name__ == "__main__":

    ticker = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2023-01-01'

    run_sma_crossover(ticker, start_date, end_date)

    run_naive_momentum(ticker, start_date, end_date)

    run_mean_reversion(ticker, start_date, end_date)
