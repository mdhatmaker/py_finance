import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f


# Define the moving average periods
FastMA = 12
SlowMA = 26

# Initialize the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()


def get_heiken_ashi(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2)
    ha_open = (rates[1]['open'] + rates[1]['close']) / 2
    ha_close = (rates[0]['open'] + rates[0]['high'] + rates[0]['low'] + rates[0]['close']) / 4
    return ha_open, ha_close


def calculate_ma(symbol, timeframe, period):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period)
    closes = [rate['close'] for rate in rates]
    return pd.Series(closes).rolling(window=period).mean().iloc[-1]


def main():
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_M1  # 1-minute timeframe

    ha_open, ha_close = get_heiken_ashi(symbol, timeframe)

    fastMA = calculate_ma(symbol, timeframe, FastMA)
    slowMA = calculate_ma(symbol, timeframe, SlowMA)

    positions = mt5.positions_get(symbol=symbol)

    if ha_close > ha_open and fastMA > slowMA:
        # Buy condition
        if not positions:
            lot = 0.1  # Define your lot size
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).ask,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script buy",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            print(result)

    elif ha_close < ha_open and fastMA < slowMA:
        # Sell condition
        if not positions:
            lot = 0.1  # Define your lot size
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": mt5.symbol_info_tick(symbol).bid,
                "deviation": 10,
                "magic": 234000,
                "comment": "python script sell",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            print(result)


if __name__ == "__main__":
    main()

# Shutdown the MetaTrader 5 terminal
mt5.shutdown()


"""
Pros:
Confirms trend direction.
Reduces false signals compared to using Heiken Ashi alone.

Cons:
Lagging indicator; slower to react to market changes.
Whipsaws in sideways markets.

Benefits:
Stronger trend confirmation.
Helps avoid trading against the trend.

Disadvantages:
Potential for delayed entries and exits.
Can miss quick, profitable moves.

Psychology: Traders need to be comfortable with the lag inherent in moving averages and maintain
confidence in the strategy during consolidation periods.
"""
