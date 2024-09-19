import yfinance as yf
import pandas as pd
import ta
import numpy as np
from datetime import datetime
from pandas import Series, DataFrame
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc


class ThreeWayAverageCrossover:

    ma_1: DataFrame
    ma_2: DataFrame
    ma_3: DataFrame
    ma_1_window: int = 3
    ma_2_window: int = 6
    ma_3_window: int = 5

    # Define the range of values for the moving average parameters
    ma1_range = range(3, 21)
    ma2_range = range(3, 21)
    ma3_range = range(3, 21)

    best_params: tuple[int, int, int]
    best_total_net_profit: float
    best_net_profit_per_trade: float

    def __init__(self):
        pass

    def moving_average_strategy(self, df, ma1, ma2, ma3):

        # Calculate the moving averages
        ma_short = df['Close'].rolling(window=ma1).mean()
        ma_mid = df['Close'].rolling(window=ma2).mean()
        ma_long = df['Close'].rolling(window=ma3).mean()

        # Find where the moving averages cross
        cross_buy = ((ma_short > ma_long) & (ma_mid > ma_long)) & ((ma_short.shift() < ma_long.shift()) | (ma_mid.shift() < ma_long.shift()))
        cross_sell = (ma_short < ma_mid) | (ma_short < ma_long)

        first_buy_signal = False
        signals = pd.DataFrame(columns=['Type', 'Date', 'Price'])

        for date, row in df.iterrows():
            if cross_buy.loc[date] and not first_buy_signal:
                first_buy_signal = True
                new_signal = pd.DataFrame({'Type': ['Buy'], 'Date': [date], 'Price': [row['Low']]})
                signals = pd.concat([signals, new_signal], ignore_index=True)
            elif cross_sell.loc[date] and first_buy_signal:
                first_buy_signal = False
                new_signal = pd.DataFrame({'Type': ['Sell'], 'Date': [date], 'Price': [row['High']]})
                signals = pd.concat([signals, new_signal], ignore_index=True)

        signals.reset_index(drop=True, inplace=True)

        # Calculate the returns for each trade
        signals['Returns'] = np.nan

        for i in range(0, len(signals) - 1, 2):
            buy_price = signals.iloc[i]['Price']
            sell_price = signals.iloc[i + 1]['Price']
            signals.iloc[i + 1, signals.columns.get_loc('Returns')] = sell_price - buy_price

        # Calculate the metrics
        total_net_profit = signals['Returns'].sum()
        losing_trade_sum = abs(signals[signals['Returns'] < 0]['Returns'].sum())
        profit_factor = signals[signals['Returns'] > 0]['Returns'].sum() / losing_trade_sum if losing_trade_sum != 0 else np.inf
        percent_profitable = len(signals[signals['Returns'] > 0]) / (len(signals) / 2) * 100
        average_trade_net_profit = signals['Returns'].mean()
        drawdown = (signals['Price'].cummax() - signals['Price']).max()

        return total_net_profit, average_trade_net_profit, profit_factor, percent_profitable, drawdown


    def find_best_parameters(self, ticker_symbol, start_date, end_date):
        # Get the data from Yahoo Finance using yfinance library
        stock_ohlc = yf.Ticker(ticker_symbol)
        df = stock_ohlc.history(start=start_date, end=end_date)

        # Initialize the best parameters and their associated metrics
        self.best_params = (0, 0, 0)
        self.best_total_net_profit = -np.inf
        self.best_net_profit_per_trade = -np.inf

        # Perform a grid search to find the best parameters
        for ma1 in self.ma1_range:
            for ma2 in self.ma2_range:
                for ma3 in self.ma3_range:
                    if ma1 != ma2 and ma1 != ma3 and ma2 != ma3:
                        total_net_profit, net_profit_per_trade, _, _, _ = self.moving_average_strategy(df, ma1, ma2, ma3)

                        if total_net_profit > self.best_total_net_profit and net_profit_per_trade > self.best_net_profit_per_trade:
                            self.best_params = (ma1, ma2, ma3)
                            self.best_total_net_profit = total_net_profit
                            self.best_net_profit_per_trade = net_profit_per_trade

        print("Best Parameters: MA1 =", self.best_params[0], "MA2 =", self.best_params[1], "MA3 =", self.best_params[2])
        print("Best Total Net Profit:", self.best_total_net_profit)
        print("Best Net Profit Per Trade:", self.best_net_profit_per_trade)


    def major_date_formatter(self, x, pos=None):
        dt = mdates.num2date(x)
        if dt.day == 1:
            return f'{dt.strftime("%b")} {dt.year}'
        return ''


    def minor_date_formatter(self, x, pos=None):
        dt = mdates.num2date(x)
        if dt.day == 1:
            return f'{dt.day}\n\n{dt.strftime("%b")} {dt.year}' if dt.month == 1 else f'{dt.day}\n\n{dt.strftime("%b")}'
        return f'{dt.day}'


    def calculate_indicator(self, ticker_symbol: str, start_date: str, end_date: str):
        # Get the data from Yahoo Finance using yfinance library
        stock_ohlc = yf.Ticker(ticker_symbol)
        df = stock_ohlc.history(start=start_date, end=end_date)

        # Calculate the moving averages
        self.ma_1 = df['Close'].rolling(window=self.ma_1_window).mean()
        self.ma_2 = df['Close'].rolling(window=self.ma_2_window).mean()
        self.ma_3 = df['Close'].rolling(window=self.ma_3_window).mean()

        # Find where the moving averages cross
        cross_buy = ((self.ma_1 > self.ma_3) & (self.ma_2 > self.ma_3)) & ((self.ma_1.shift() < self.ma_3.shift()) | (self.ma_2.shift() < self.ma_3.shift()))
        cross_sell = (self.ma_1 < self.ma_2) | (self.ma_1 < self.ma_3)

        # Convert the date to the matplotlib date format
        df['Date'] = mdates.date2num(df.index)

        return df, cross_buy, cross_sell


    def plot_threeway_average_crossover(self, ticker, df, cross_buy, cross_sell):
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot the moving averages
        ax.plot(df.index, self.ma_1, color='tab:blue', label='{0}-day MA'.format(self.ma_1_window), linestyle='--')
        ax.plot(df.index, self.ma_2, color='tab:orange', label='{0}-day MA'.format(self.ma_2_window), linestyle='--')
        ax.plot(df.index, self.ma_3, color='tab:green', label='{0}-day MA'.format(self.ma_3_window), linestyle='--')

        # Plot the candlesticks
        candlestick_ohlc(ax, df[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.6, colorup='green', colordown='red')

        # Create markers for buy and sell signals
        buy_marker = mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10, label='Buy')
        sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10, label='Sell')

        # Initialize buy and sell signal flags
        first_buy_signal = False
        buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []

        # Record prices and dates for buy and sell signals
        for date, row in df.iterrows():
            if cross_buy.loc[date] and not first_buy_signal:
                first_buy_signal = True
                buy_dates.append(date)
                buy_prices.append(row['Low'])
            elif cross_sell.loc[date] and first_buy_signal:
                first_buy_signal = False
                sell_dates.append(date)
                sell_prices.append(row['High'])

        # Plot the buy and sell markers
        ax.plot(buy_dates, buy_prices, '^', markersize=7, color='blue', label='Buy')
        ax.plot(sell_dates, sell_prices, 'v', markersize=7, color='red', label='Sell')

        # Add the legend and title
        ax.legend(handles=[buy_marker, sell_marker])
        ax.set_title(ticker + ' Stock Price Chart with Buy and Sell markers')

        # Set y-axis label
        ax.set_ylabel('Price')

        plt.show()


def run_threeway_average_crossover(ticker_symbol: str, start_date: str, end_date: str):
    ind = ThreeWayAverageCrossover()
    ind.find_best_parameters(ticker_symbol, start_date, end_date)
    df, cross_buy, cross_sell = ind.calculate_indicator(ticker_symbol, start_date, end_date)
    ind.plot_threeway_average_crossover(ticker_symbol, df, cross_buy, cross_sell)


if __name__ == "__main__":

    ticker = 'UNA.AS'
    start_date = '2020-01-01'
    end_date = '2024-03-26'

    ind = ThreeWayAverageCrossover()
    ind.find_best_parameters(ticker, start_date, end_date)
    df, cross_buy, cross_sell = ind.calculate_indicator(ticker, start_date, end_date)
    ind.plot_threeway_average_crossover(ticker, df, cross_buy, cross_sell)
