import MetaTrader5 as mt5
import pandas as pd

# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
k_period = 14
d_period = 3
slowing = 3
symbol = "EURUSD"
lots = 0.1

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to calculate Stochastic Oscillator
def stochastic(df, k_period, d_period, slowing):
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    k_value = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    k_value_slowed = k_value.rolling(window=slowing).mean()
    d_value = k_value_slowed.rolling(window=d_period).mean()
    return k_value_slowed, d_value

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

# Calculate Stochastic Oscillator
k_value, d_value = stochastic(df, k_period, d_period, slowing)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]
k_latest = k_value.iloc[-1]
d_latest = d_value.iloc[-1]

# Check conditions
if ha_close > ha_open and k_latest < 20 and k_latest > d_latest:
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
elif ha_close < ha_open and k_latest > 80 and k_latest < d_latest:
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
Combines trend following with momentum oscillation.
Filters out weaker signals.

Cons:
May miss some trending opportunities.
Stochastic can give false signals in strong trends.

Benefits:
Helps avoid buying at tops and selling at bottoms.
Reduces the number of trades, focusing on higher probability setups.

Disadvantages:
Potential for fewer trades.
Stochastic can be lagging during strong trends.

Psychology: Traders need patience to wait for the best setups and must avoid the temptation to
enter trades prematurely.
"""
