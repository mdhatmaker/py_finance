import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from api.utils.yahoo_finance import download_ohlc

# https://towardsdatascience.com/a-simple-breakout-trading-strategy-in-python-eb043b868d8e


def donchian(df: DataFrame, period=20):   # low, high, where_up, where_down, median, period=20):
    df['donch_up'] = np.nan
    df['donch_down'] = np.nan
    df['median'] = np.nan
    for i in range(len(df)):
        if i >= period:
            df.iloc[i, 6] = df.iloc[i - period:i + 1, 2].max()      # max of column 'High'
            df.iloc[i, 7] = df.iloc[i - period:i + 1, 1].min()      # min of column 'Low'
    df['median'] = (df['donch_up'] + df['donch_down']) / 2

    # for i in range(len(data)):
    #     if i >= period:
    #         data[i]['donch_up'] = max(data[i - period:i + 1]['High'])
    #     # try:
    #     #     data[i, where_up] = max(data[i - period:i + 1, 1])
    #     # except ValueError:
    #     #     pass
    # for i in range(len(data)):
    #     if i >= period:
    #         data[i]['donch_down'] = min(data[i - period:i + 1]['Low'])
    #     # try:
    #     #     data[i, where_down] = min(data[i - period:i + 1, 2])
    #     # except ValueError:
    #     #     pass
    #
    # for i in range(len(data)):
    #     data[i]['median'] = (data[i]['donch_up'] + data[i]['donch_down']) / 2
    #     # try:
    #     #     data[i]['median'] = (data[i, where_up] + data[i, where_down]) / 2
    #     # except ValueError:
    #     #     pass

    return df.dropna()


def plot_donchian_breakout(ticker, df):
    plt.figure(figsize=(12, 8))
    plt.plot(df.iloc[-500:, 3], color='black')
    plt.plot(df.iloc[-500:, 6])         # , ['donch_up'])
    plt.plot(df.iloc[-500:, 7])         # , ['donch_down'])
    plt.plot(df.iloc[-500:, 8])         # , ['median'])
    plt.xlabel('Date')
    plt.ylabel(f'{ticker} Price')
    plt.title(f'{ticker} Donchian Breakout Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()


def donchian_signals(df: DataFrame):
    df['Buy_Signal'] = np.nan
    df['Sell_Signal'] = np.nan
    for i in range(1, len(df)):
        if df.iloc[i, 4] > df.iloc[i - 1, 6]:       # compare column 'Close' to column 'donch_up'
            df.iloc[i, 9] = 1                       # set column 'Buy_Signal' to 1
        elif df.iloc[i, 4] < df.iloc[i - 1, 7]:     # compare column 'Close' to column 'donch_down'
            df.iloc[i, 10] = -1                     # set column 'Sell_Signal' to -1

    return df


def run_donchian_breakout(ticker, start_date, end_date=None):
    data = download_ohlc(ticker, start_date, end_date)
    df = donchian(data, 20)  #  2, 1, 4, 5, 6, 20)
    df  = donchian_signals(df)
    plot_donchian_breakout(ticker, df)


if __name__ == "__main__":
    run_donchian_breakout('NVDA', '2020-01-01', None)


"""
When markets are strongly trending, mean reversion strategies tend to fail and therefore we always
have to adapt our market approach accordingly.
"""
