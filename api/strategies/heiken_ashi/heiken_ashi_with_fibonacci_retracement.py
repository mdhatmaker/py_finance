import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
fib_retracement_level = 0.618
symbol = "EURUSD"
lots = 0.1

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to calculate Fibonacci retracement level
def calculate_fib_retracement(df, level):
    recent_high = df['high'].max()
    recent_low = df['low'].min()
    retracement = recent_high - (recent_high - recent_low) * level
    return retracement

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

# Calculate Fibonacci retracement level
fib_retracement = calculate_fib_retracement(df, fib_retracement_level)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]

# Check conditions
if ha_close > ha_open and ha_close > fib_retracement:
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
                "comment": "Buy with Fibonacci retracement condition",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )
elif ha_close < ha_open and ha_close < fib_retracement:
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
                "comment": "Sell with Fibonacci retracement condition",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )

# Shutdown MT5 connection
mt5.shutdown()


"""
Pros:
Combines trend following with key retracement levels.
Provides strong entry points.

Cons:
May miss some trending opportunities.
Fibonacci levels can be subjective.

Benefits:
Helps identify strong bounce opportunities.
Reduces the number of trades, focusing on higher probability setups.

Disadvantages:
Potential for fewer trades.
Fibonacci levels can be difficult to identify accurately.

Psychology: Traders need to be patient to wait for clear bounces and disciplined to avoid
entering trades prematurely.
"""

