import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
step = 0.02
maximum = 0.2
symbol = "EURUSD"
lots = 0.1

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to calculate Parabolic SAR
def parabolic_sar(df, step, maximum):
    sar = pd.Series([0.0] * len(df), index=df.index)
    af = step
    ep = df['low'][0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    sar[0] = df['low'][0] - (step * (df['high'][0] - df['low'][0]))

    for i in range(1, len(df)):
        if trend == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if df['high'][i] > ep:
                ep = df['high'][i]
                af = min(af + step, maximum)
            if df['low'][i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = df['low'][i]
                af = step
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            if df['low'][i] < ep:
                ep = df['low'][i]
                af = min(af + step, maximum)
            if df['high'][i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = df['high'][i]
                af = step

    return sar

# Function to get data
def get_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from(symbol, timeframe, mt5.TIMEFRAME_M1, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Fetching the data
bars = 100
df = get_data(symbol, mt5.TIMEFRAME_M1, bars)

# Calculate Heiken Ashi
ha_df = heiken_ashi(df)

# Calculate Parabolic SAR
par_sar = parabolic_sar(df, step, maximum)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]
par_sar_value = par_sar.iloc[-1]

# Check conditions
if ha_close > ha_open and par_sar_value < ha_close:
    # Buy condition
    if not mt5.positions_get(symbol=symbol):
        mt5.order_send(
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).ask,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )
elif ha_close < ha_open and par_sar_value > ha_close:
    # Sell condition
    if not mt5.positions_get(symbol=symbol):
        mt5.order_send(
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )

# Shutdown MT5 connection
mt5.shutdown()


"""
Pros:
Combines trend and reversal indicators.
Helps identify strong trends and potential reversals.

Cons:
Whipsaws in sideways markets.
Parabolic SAR can be lagging in strong trends.

Benefits:
Provides clear entry and exit points.
Helps traders stay in trades during strong trends.

Disadvantages:
Potential for delayed entries and exits.
Parabolic SAR can be prone to false signals in choppy markets.

Psychology: Traders must be disciplined to follow the signals, especially during potential
reversals, and avoid the temptation to exit trades prematurely.
"""
