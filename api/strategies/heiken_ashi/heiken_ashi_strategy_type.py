import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Enums for strategy types
class StratType:
    SIMPLE_STRAT = 0
    PREDICT_STRAT = 1

# Input parameters
inp_strat_type = StratType.SIMPLE_STRAT

# Function to calculate Heiken Ashi
def heiken_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (ha_df['ha_close'].shift(1) + ha_df['ha_open'].shift(1)) / 2
    ha_df['ha_open'].fillna((df['open'] + df['close']) / 2, inplace=True)
    return ha_df

# Function to get data
def get_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from(symbol, timeframe, mt5.TIMEFRAME_M1, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Placeholder functions for different strategy types
def simple_strategy():
    pass

def predict_strategy():
    pass

# Fetching the data
symbol = "EURUSD"
bars = 100
df = get_data(symbol, mt5.TIMEFRAME_M1, bars)

# Calculate Heiken Ashi
ha_df = heiken_ashi(df)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]

# Main trading logic based on selected strategy type
def main():
    if inp_strat_type == StratType.SIMPLE_STRAT:
        simple_strategy()
    elif inp_strat_type == StratType.PREDICT_STRAT:
        predict_strategy()

    # Example of using Heiken Ashi values for a basic strategy
    if ha_close > ha_open:
        # Buy condition
        if not mt5.positions_get(symbol=symbol):
            mt5.order_send(
                request={
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": 0.1,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(symbol).ask,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Buy condition met",
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
                    "volume": 0.1,
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": mt5.symbol_info_tick(symbol).bid,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "Sell condition met",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
            )

# Execute main function
main()

# Shutdown MT5 connection
mt5.shutdown()


