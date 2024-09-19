import MetaTrader5 as mt5
import pandas as pd

# https://medium.com/@jsgastoniriartecabrera/heikin-ashi-strategies-02a62f10754f

"""
Heikin-Ashi bars are calculated using the following formulas:
- Heikin-Ashi Close (haclose): `(open + high + low + close) / 4`
- Heikin-Ashi Open (haopen): `previous haopen + previous haclose / 2` (initialized as `(open + close) / 2` if no previous value)
- Heikin-Ashi High (hahigh): `max(high, max(haopen, haclose))`
- Heikin-Ashi Low (halow): `min(low, min(haopen, haclose))`
"""


# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Input parameters
symbol = "EURUSD"
lots = 0.1

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
bars = 100
df = get_data(symbol, mt5.TIMEFRAME_M1, bars)

# Calculate Heiken Ashi
ha_df = heiken_ashi(df)

# Get the latest values
ha_open = ha_df['ha_open'].iloc[-1]
ha_close = ha_df['ha_close'].iloc[-1]

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
                "deviation": 10,
                "magic": 234000,
                "comment": "python script order",
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
# Heikin-Ashi Trend Following (MQL5 code)

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   double haOpen, haClose;
   haOpen = iCustom(NULL, 0, "Heiken Ashi", 0, 1);
   haClose = iCustom(NULL, 0, "Heiken Ashi", 3, 1);

   if(haClose > haOpen)
     {
      // Buy condition
      if(PositionSelect(Symbol()) == false)
        {
         trade.Buy(lots);
        }
     }
   else if(haClose < haOpen)
     {
      // Sell condition
      if(PositionSelect(Symbol()) == false)
        {
         trade.Sell(lots);
        }
     }
  }
  
  
Pros:
Simple and easy to understand.
Effective in trending markets.

Cons:
Whipsaw losses in sideways markets.
Delayed entries and exits due to smoothing.

Benefits:
Reduces noise, making trend identification easier.
Helps traders stay in trends longer.

Disadvantages:
Can miss short-term reversals.
Potentially large drawdowns during consolidation.

Psychology: Trend following requires patience and discipline to let profits run and cut losses short.
Traders need to trust the method even during periods of drawdowns.
"""

"""
2. Heiken Ashi and Moving Average Crossover (MQL5 code)
Explanation: Combining Heiken Ashi with a Moving Average Crossover can help confirm trend direction. Buy when Heiken Ashi candles are green and the fast MA crosses above the slow MA. Sell when Heiken Ashi candles are red and the fast MA crosses below the slow MA.
MQL5 Code:
input int FastMA = 12;
input int SlowMA = 26;

//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
void OnTick()
  {
   double haOpen, haClose;
   haOpen = iCustom(NULL, 0, "Heiken Ashi", 0, 1);
   haClose = iCustom(NULL, 0, "Heiken Ashi", 3, 1);
   
   double fastMA = iMA(NULL, 0, FastMA, 0, MODE_SMA, PRICE_CLOSE, 0);
   double slowMA = iMA(NULL, 0, SlowMA, 0, MODE_SMA, PRICE_CLOSE, 0);

   if(haClose > haOpen && fastMA > slowMA)
     {
      // Buy condition
      if(PositionSelect(Symbol()) == false)
        {
         trade.Buy(lots);
        }
     }
   else if(haClose < haOpen && fastMA < slowMA)
     {
      // Sell condition
      if(PositionSelect(Symbol()) == false)
        {
         trade.Sell(lots);
        }
     }
  }
  
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

