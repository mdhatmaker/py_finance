import ccxt
import warnings
from matplotlib.pyplot import fill_between
import pandas as pd
import numpy as np
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# https://quantitativepy.substack.com/p/mastering-market-trends-with-python
# pip install ccxt pandas matplotlib pandas-ta mplfinance
"""
About the Supertrend Indicator:
Imagine a river that sometimes flows smoothly and at other times behaves erratically, even breaking its banks.
This mirrors market behavior—prices can move smoothly or swing wildly, spiking unpredictably. The Supertrend
indicator measures these swings, known as volatility, using the Average True Range (ATR). The ATR, representing
the trading range of an asset over a specified period, helps generate the upper and lower bands—boundaries within
which the asset prices fluctuate. A price break above the upper boundary signals an uptrend, while a break below
the lower boundary signals a downtrend.

Parameter Sensitivity:
The supertrend indicator can be sensitive to its parameters such as the period, and the atr volatility multiplier.
Different trading pairs can move around a lot in a given period, but be pretty quiet on others. So what worked well for
one pair might not work as well for a different one. You could tweak the volatility parameters to suit the trading pair.

Volatility Differences:
The SuperTrend indicator relies heavily on the ATR to capture trends, but if a trading pair is so volatile or experiences
frequent sharp price fluctuations, it may produce false signals or fail to capture meaningful trends. Therefore the
supertrend indicator may not be suitable for highly volatile assets.
"""


def fetch_asset_data(symbol, start_date, interval, exchange):
    start_date_ms = exchange.parse8601(start_date)

    ohlcv = exchange.fetch_ohlcv(symbol, interval, since=start_date_ms)

    header = ["date", "open", "high", "low", "close", "volume"]

    df = pd.DataFrame(ohlcv, columns=header)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index("date", inplace=True)
    df.drop(df.index[-1], inplace=True)

    return df


# Calculate SuperTrend lines
def supertrend(df, atr_multiplier=3):
    current_average_high_low = (df['high']+df['low'])/2
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], period=15)
    df.dropna(inplace=True)

    df['basicUpperband'] = current_average_high_low + (atr_multiplier * df['atr'])

    df['basicLowerband'] = current_average_high_low - (atr_multiplier * df['atr'])

    upperBand, lowerBand = [df['basicUpperband'].iloc[0]], [df['basicLowerband'].iloc[0]]

    for i in range(1, len(df)):
        upperBand.append(min(df['basicUpperband'].iloc[i], upperBand[-1]) if df['close'].iloc[i-1] <= upperBand[-1] else df['basicUpperband'].iloc[i])

        lowerBand.append(max(df['basicLowerband'].iloc[i], lowerBand[-1]) if df['close'].iloc[i-1] >= lowerBand[-1] else df['basicLowerband'].iloc[i])

    df['upperband'] = upperBand
    df['lowerband'] = lowerBand
    df.drop(['basicUpperband', 'basicLowerband'], axis=1, inplace=True)

    return df


# Generate signals
def generate_signals(df):
    signals = [0]

    for i in range(1, len(df)):
        if df['close'][i] > df['upperband'][i]:
            signals.append(1)

        elif df['close'][i] < df['lowerband'][i]:
            signals.append(-1)

        else:
            signals.append(signals[i-1])

    df['signals'] = signals

    df['signals'] = df["signals"].shift(1)

    return df


# Generate positions
def create_positions(df):
    df['upperband'][df['signals'] == 1] = np.nan
    df['lowerband'][df['signals'] == -1] = np.nan

    buy_positions, sell_positions = [np.nan], [np.nan]

    for i in range(1, len(df)):
        if df['signals'][i] == 1 and df['signals'][i] != df['signals'][i-1]:
            buy_positions.append(df['close'][i])
            sell_positions.append(np.nan)
        elif df['signals'][i] == -1 and df['signals'][i] != df['signals'][i-1]:
            sell_positions.append(df['close'][i])
            buy_positions.append(np.nan)
        else:
            buy_positions.append(np.nan)
            sell_positions.append(np.nan)

    df['buy_positions'] = buy_positions

    df['sell_positions'] = sell_positions

    return df


# Visualize the data
def plot_data(df, symbol):
    lowerband_line = mpf.make_addplot(df['lowerband'], label="lowerband", color='green')

    upperband_line = mpf.make_addplot(df['upperband'], label="upperband", color='red')

    buy_position_makers = mpf.make_addplot(df['buy_positions'], type='scatter', marker='^', label="Buy", markersize=80, color='#2cf651')

    sell_position_makers = mpf.make_addplot(df['sell_positions'], type='scatter', marker='v', label="Sell", markersize=80, color='#f50100')

    apd = [lowerband_line, upperband_line, buy_position_makers, sell_position_makers]

    lowerband_fill = dict(y1=df['close'].values, y2=df['lowerband'].values, panel=0, alpha=0.3, color="#CCFFCC")

    upperband_fill = dict(y1=df['close'].values, y2=df['upperband'].values, panel=0, alpha=0.3, color="#FFCCCC")

    fills = [lowerband_fill, upperband_fill]

    mpf.plot(df, addplot=apd, type='candle', volume=True, style='charles', xrotation=20, title=str(symbol + ' Supertrend Plot'), fill_between=fills)


# Calculate strategy performance
def strategy_performance(strategy_df, capital=100, leverage=1):
    cumulative_balance = capital
    investment = capital
    pl = 0
    max_drawdown = 0
    max_drawdown_percentage = 0
    balance_list = [capital]
    pnl_list = [0]
    investment_list = [capital]
    peak_balance = capital

    for index in range(1, len(strategy_df)):
        row = strategy_df.iloc[index]

        if row['signals'] == 1:
            pl = ((row['close'] - row['open']) / row['open']) * investment * leverage
        elif row['signals'] == -1:
            pl = ((row['open'] - row['close']) / row['close']) * investment * leverage
        else:
            pl = 0

        if row['signals'] != strategy_df.iloc[index - 1]['signals']:
            investment = cumulative_balance

        cumulative_balance += pl
        investment_list.append(investment)
        balance_list.append(cumulative_balance)

        pnl_list.append(pl)

        drawdown = cumulative_balance - peak_balance

        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_percentage = (max_drawdown / peak_balance) * 100

        if cumulative_balance > peak_balance:
            peak_balance = cumulative_balance

    strategy_df['investment'] = investment_list
    strategy_df['cumulative_balance'] = balance_list
    strategy_df['pl'] = pnl_list
    strategy_df['cumPL'] = strategy_df['pl'].cumsum()

    print(f'Initial Capital: ${capital}')

    print(f'Total P&L: ${strategy_df["cumPL"].iloc[-1]}')

    print(f'Cumulative Balance: ${strategy_df["cumulative_balance"].iloc[-1]}')

    print(f'Max Drawdown: ${max_drawdown} ({max_drawdown_percentage}%)')


def run_supertrend_indicator(symbol, start_date, interval, exchange):
    # Initialize data fetch parameters
    # symbol = "BTC/USDT"
    # start_date = "2024-1-1"
    # interval = '4h'
    # exchange = ccxt.binance()

    # Fetch historical OHLC data for ETH/USDT
    data = fetch_asset_data(symbol=symbol, start_date=start_date, interval=interval, exchange=exchange)

    volatility = 3

    # Apply Supertrend formula
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)

    # Generate the Signals
    supertrend_positions = generate_signals(supertrend_data)

    # Generate the Positions
    supertrend_positions = create_positions(supertrend_positions)

    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100, leverage=1)
    print(supertrend_df)

    # Plot data
    plot_data(supertrend_positions, symbol=symbol)

    # Plot the performance curve
    # plot_performance_curve(supertrend_df)


if __name__ == '__main__':

    symbol = "BTC/USDT"
    start_date = "2024-1-1"
    interval = '4h'
    exchange = ccxt.binance()
    run_supertrend_indicator(symbol, start_date, interval, exchange)


