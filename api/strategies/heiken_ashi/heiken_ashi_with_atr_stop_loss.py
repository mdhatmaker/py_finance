import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
atr_multiplier = 1.5
atr_period = 14
symbol = "EURUSD"
lots = 0.1

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to calculate ATR
def atr(df, period):
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])), axis=1
    )
    atr = df['tr'].rolling(window=period).mean()
    return atr

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

# Calculate ATR
atr_values = atr(df, atr_period)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]
atr_value = atr_values.iloc[-1]

# Check conditions
if ha_close > ha_open:
    # Buy condition
    if not mt5.positions_get(symbol=symbol):
        mt5.order_send(
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).ask,
                "sl": mt5.symbol_info_tick(symbol).bid - atr_multiplier * atr_value,
                "tp": 0,
                "deviation": 10,
                "magic": 234000,
                "comment": "Buy with ATR stop loss",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )
elif ha_close < ha_open:
    # Sell condition
    if not mt5.positions_get(symbol=symbol):
        mt5.order_send(
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).bid,
                "sl": mt5.symbol_info_tick(symbol).ask + atr_multiplier * atr_value,
                "tp": 0,
                "deviation": 10,
                "magic": 234000,
                "comment": "Sell with ATR stop loss",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )

# Shutdown MT5 connection
mt5.shutdown()


"""
Pros:
Combines trend following with volatility-based stop loss.
Helps protect against large losses.

Cons:
May result in stop loss being hit frequently in volatile markets.
Can reduce profit potential by tightening stops too much.

Benefits:
Provides clear stop loss levels based on market volatility.
Helps manage risk effectively.

Disadvantages:
Potential for frequent stop-outs in volatile conditions.
ATR-based stops may be too tight in trending markets.

Psychology: Traders need to accept the possibility of frequent stop-outs and maintain discipline to
re-enter trades when conditions align.
"""

