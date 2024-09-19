from itertools import combinations
from statsmodels.tsa.stattools import coint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
import numpy as np
import os

from api.utils.yahoo_finance import load_ticker_ts_df


# https://medium.com/the-investors-handbook/using-z-score-in-trading-a-python-study-ae90e3fec6ac


# The function to add a certain number of columns
def adder(Data, times):
    for i in range(1, times + 1):
        z = np.zeros((len(Data), 1), dtype=float)
        Data = np.append(Data, z, axis=1)
    return Data


# The function to deleter a certain number of columns
def deleter(Data, index, times):
    for i in range(1, times + 1):
        Data = np.delete(Data, index, axis=1)

    return Data


# The function to delete a certain number of rows from the beginning
def jump(Data, jump):
    Data = Data[jump:, ]

    return Data


def normalizer(Data, lookback, what, where):
    for i in range(len(Data)):

        try:
            Data[i, where] = (Data[i, what] - min(Data[i - lookback + 1:i + 1, what])) / (
                        max(Data[i - lookback + 1:i + 1, what]) - min(Data[i - lookback + 1:i + 1, what]))

        except ValueError:
            pass

    Data[:, where] = Data[:, where] * 100
    Data = jump(Data, lookback)
    return Data


def z_score_indicator(Data, ma_lookback, std_lookback, close, where):
    mu = Data.Close.rolling(ma_lookback).mean()
    sig = Data.Close.rolling(std_lookback).std()
    z = (Data-mu) / sig
    Data['z'] = z
    return Data
    # # Adding Columns
    # Data = adder(Data, 1)
    #
    # # Calculating the moving average
    # Data = ma(Data, ma_lookback, close, where)
    #
    # # Calculating the standard deviation
    # Data = volatility(Data, std_lookback, close, where + 1)
    #
    # # Calculating the Z-Score
    # for i in range(len(Data)):
    #     Data[i, where + 2] = (Data[i, close] - Data[i, where]) / Data[i, where + 1]
    #
    # # Cleaning
    # Data = deleter(Data, where, 2)
    #
    # return Data


def performance(indexer, Data, name):
    # Profitability index
    indexer = np.delete(indexer, 0, axis=1)
    indexer = np.delete(indexer, 0, axis=1)

    profits = []
    losses = []
    np.count_nonzero(Data[:, 7])
    np.count_nonzero(Data[:, 8])

    for i in range(len(indexer)):

        if indexer[i, 0] > 0:
            value = indexer[i, 0]
            profits = np.append(profits, value)

        if indexer[i, 0] < 0:
            value = indexer[i, 0]
            losses = np.append(losses, value)

    # Hit ratio calculation
    hit_ratio = round((len(profits) / (len(profits) + len(losses))) * 100, 2)

    #realized_risk_reward = round(abs(profits.mean() / losses.mean()), 2)
    realized_risk_reward = round(np.abs(np.mean(profits)) / np.mean(losses), 2)

    # Expected and total profits / losses
    expected_profits = np.mean(profits)
    expected_losses = np.abs(np.mean(losses))
    total_profits = round(np.sum(profits), 3)
    total_losses = round(np.abs(np.sum(losses)), 3)

    # Expectancy
    expectancy = round((expected_profits * (hit_ratio / 100)) \
                       - (expected_losses * (1 - (hit_ratio / 100))), 2)

    # Largest Win and Largest Loss
    largest_win = round(max(profits), 2)
    largest_loss = round(min(losses), 2)
    # Total Return
    indexer = Data[:, 10:12]

    # Creating a combined array for long and short returns
    z = np.zeros((len(Data), 1), dtype=float)
    indexer = np.append(indexer, z, axis=1)

    # Combining Returns
    for i in range(len(indexer)):
        try:
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot)

            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot)
        except IndexError:
            pass

    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot

    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis=1)
    indexer[:, 3] = investment

    # Adding returns to the balance
    for i in range(len(indexer)):
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])

    indexer = np.array(indexer)

    total_return = (indexer[-1, 3] / indexer[0, 3]) - 1
    total_return = total_return * 100

    print('-----------Performance-----------', name)
    print('Hit ratio       = ', hit_ratio, '%')
    print('Net profit      = ', '$', round(indexer[-1, 3] - indexer[0, 3], 2))
    print('Expectancy      = ', '$', expectancy, 'per trade')
    print('Profit factor   = ', round(total_profits / total_losses, 2))
    print('Total Return    = ', round(total_return, 2), '%')
    print('')
    print('Average Gain    = ', '$', round((expected_profits), 2), 'per trade')
    print('Average Loss    = ', '$', round((expected_losses * -1), 2), 'per trade')
    print('Largest Gain    = ', '$', largest_win)
    print('Largest Loss    = ', '$', largest_loss)
    print('')
    print('Realized RR     = ', realized_risk_reward)
    print('Minimum         =', '$', round(min(indexer[:, 3]), 2))
    print('Maximum         =', '$', round(max(indexer[:, 3]), 2))
    print('Trades          =', len(profits) + len(losses))


def signal(Data, what, buy, sell, lower_barrier = -2, upper_barrier = 2):
    Data = adder(Data, 10)

    for i in range(len(Data)):

        if Data[i, what] <= lower_barrier and Data[i - 1, buy] == 0 and Data[i - 2, buy] == 0 and Data[i - 3, buy] == 0 and Data[
            i - 4, buy] == 0:
            Data[i, buy] = 1

        elif Data[i, what] >= upper_barrier and Data[i - 1, sell] == 0 and Data[i - 2, sell] == 0 and Data[i - 3, sell] == 0 and Data[
            i - 4, sell] == 0:
            Data[i, sell] = -1

    return Data


"""
The first step into building the Equity Curve is to calculate the profits and losses from the individual trades
we are taking. For simplicity reasons, we can consider buying and selling at closing prices. This means that when
we get the signal from the indicator or the pattern on close, we initiate the trade on the close until getting
another signal where we exit and initiate the new trade.

This function will give us columns 8 and 9 populated with the gross profit and loss results from the trades taken.
"""
def holding(Data, buy, sell, buy_return, sell_return):
    for i in range(len(Data)):
        try:
            if Data[i, buy] == 1:
                for a in range(i + 1, i + 1000):
                    if Data[a, buy] != 0 or Data[a, sell] != 0:
                        Data[a, buy_return] = (Data[a, 3] - Data[i, 3])
                        break
                else:
                    continue

            elif Data[i, sell] == -1:
                for a in range(i + 1, i + 1000):
                    if Data[a, buy] != 0 or Data[a, sell] != 0:
                        Data[a, sell_return] = (Data[i, 3] - Data[a, 3])
                        break
                    else:
                        continue

        except IndexError:
            pass

    return Data


def indexer(Data, expected_cost, lot, investment):
    # Charting portfolio evolution
    indexer = Data[:, 8:10]

    # Creating a combined array for long and short returns
    z = np.zeros((len(Data), 1), dtype=float)
    indexer = np.append(indexer, z, axis=1)

    # Combining Returns
    for i in range(len(indexer)):
        try:
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot)

            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot)
        except IndexError:
            pass

    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot

    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis=1)
    indexer[:, 3] = investment

    # Adding returns to the balance
    for i in range(len(indexer)):
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])

    indexer = np.array(indexer)

    return np.array(indexer)



if __name__ == "__main__":

    ticker = 'EURUSD=X'
    start_date = '2017-01-01'
    end_date = None
    my_data = load_ticker_ts_df(ticker, start_date, end_date)

    # lower_barrier = -2
    # upper_barrier = 2
    my_data = signal(my_data, 4, 6, 7)
    my_data = holding(my_data, 6, 7, 8, 9)

    # Using the function for a 0.1 lot strategy on $10,000 investment
    lot = 10000
    expected_cost = 0.5 * (lot / 10000)  # 0.5 pip spread
    investment = 10000
    equity_curve = indexer(my_data, expected_cost, lot, investment)

    performance(equity_curve, my_data, ticker)


    # We can derive contrarian trading rules from the definition of the indicator such as the following:
    # Go long (Buy) whenever the 21-period Z-score reaches -2.0.
    # Go short (Sell) whenever the 21-period Z-score reaches 2.0

    # Calculating the 21-period Z-score
    my_data = z_score_indicator(my_data, 21, 21, 3, 4)





