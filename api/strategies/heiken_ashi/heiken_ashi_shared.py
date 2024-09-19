import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Enums for strategy types
class StrategyType:
    TREND = 0
    CCI = 1
    VOLUME = 2
    S_P = 3
    FIBO = 4
    MAs = 5
    RSI = 6
    BOL = 7
    MACD = 8
    STOCH = 9
    PSAR = 10

# Enums for lot types
class LotType:
    FIX = 0
    RISK = 1

# Enums for stop types
class StopsType:
    ATR = 0
    PIPs = 1
    PIP_VOL = 2
    ATR_VOL = 3
    ADX_PIP = 4
    ADX_ATR = 5
    ADX_PIP_VOL = 6
    ADX_ATR_VOL = 7
    WO = 8

# Input parameters
inp_strategy_type = StrategyType.TREND
inp_lot_type = LotType.RISK
inp_lot_fix = 0.1
inp_lot_risk = 2.0
inp_stops_type = StopsType.PIPs

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

# Fetching the data
symbol = "EURUSD"
bars = 100
df = get_data(symbol, mt5.TIMEFRAME_M1, bars)

# Calculate Heiken Ashi
ha_df = heiken_ashi(df)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]

# Placeholder functions for different strategies and stops
def strategy_trend():
    pass

def strategy_cci():
    pass

# Add other strategy functions here...

def stops_atr():
    pass

def stops_pips():
    pass

# Add other stops functions here...

# Determine lot size based on type
def calculate_lot_size():
    if inp_lot_type == LotType.FIX:
        return inp_lot_fix
    elif inp_lot_type == LotType.RISK:
        # Implement risk-based lot calculation here
        return inp_lot_risk
    return 0

# Main trading logic based on selected strategy and stops
def main():
    lot_size = calculate_lot_size()

    if inp_strategy_type == StrategyType.TREND:
        strategy_trend()
    elif inp_strategy_type == StrategyType.CCI:
        strategy_cci()
    # Add other strategy conditions here...

    if inp_stops_type == StopsType.ATR:
        stops_atr()
    elif inp_stops_type == StopsType.PIPs:
        stops_pips()
    # Add other stops conditions here...

    # Example of using Heiken Ashi values for a basic strategy
    if ha_close > ha_open:
        # Buy condition
        if not mt5.positions_get(symbol=symbol):
            mt5.order_send(
                request={
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
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
                    "volume": lot_size,
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


