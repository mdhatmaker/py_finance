import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
bb_period = 20
bb_deviation = 2.0
symbol = "EURUSD"
lots = 0.1

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to calculate Bollinger Bands
def bollinger_bands(series, period, deviation):
    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()
    upper_band = middle_band + (std_dev * deviation)
    lower_band = middle_band - (std_dev * deviation)
    return upper_band, middle_band, lower_band

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

# Calculate Bollinger Bands
upper_band, middle_band, lower_band = bollinger_bands(df['close'], bb_period, bb_deviation)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]
middle_band_value = middle_band.iloc[-1]

# Check conditions
if ha_close > ha_open and ha_close > middle_band_value:
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
elif ha_close < ha_open and ha_close < middle_band_value:
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
Combines trend and volatility indicators.
Helps identify strong trends.

Cons:
May give late signals.
Whipsaws in non-trending markets.

Benefits:
Provides clear entry points.
Helps traders stay in trades during strong trends.

Disadvantages:
Potential for delayed entries and exits.
Bollinger Bands can be lagging indicators.

Psychology: Traders must have the discipline to wait for clear signals and the patience to hold
positions during volatile market conditions.
"""


