import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
adx_period = 14
adx_threshold = 25.0
symbol = "EURUSD"
lots = 0.1


# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df


# Function to calculate ADX
def adx(df, period):
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])), axis=1
    )
    df['dm_plus'] = df['high'].diff()
    df['dm_minus'] = df['low'].diff()
    df['dm_plus'] = df['dm_plus'].apply(lambda x: x if x > 0 else 0)
    df['dm_minus'] = df['dm_minus'].apply(lambda x: -x if x < 0 else 0)

    tr_smma = df['tr'].rolling(window=period).mean()
    dm_plus_smma = df['dm_plus'].rolling(window=period).mean()
    dm_minus_smma = df['dm_minus'].rolling(window=period).mean()

    df['di_plus'] = 100 * (dm_plus_smma / tr_smma)
    df['di_minus'] = 100 * (dm_minus_smma / tr_smma)

    df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    adx = df['dx'].rolling(window=period).mean()

    return adx


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

# Calculate ADX
adx_values = adx(df, adx_period)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]
adx_value = adx_values.iloc[-1]

# Check conditions
if ha_close > ha_open and adx_value > adx_threshold:
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
                "comment": "Buy with ADX condition",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )
elif ha_close < ha_open and adx_value > adx_threshold:
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
                "comment": "Sell with ADX condition",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
        )

# Shutdown MT5 connection
mt5.shutdown()


"""
Pros:
Combines trend following with trend strength confirmation.
Helps avoid weak trends.

Cons:
May miss some trending opportunities.
ADX can be lagging in strong trends.

Benefits:
Filters out weak trends and consolidations.
Provides stronger trend confirmation.

Disadvantages:
Potential for fewer trades.
ADX can be slow to react to new trends.

Psychology: Traders need patience to wait for strong trend confirmations and discipline to avoid 
rading in weak market conditions.
"""

