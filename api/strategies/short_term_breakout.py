import yfinance as yf
from api.utils.yahoo_finance import download_ohlc
# from ib_insync import IB, Stock

# https://medium.com/@thornexdaniel/creating-a-short-term-breakout-strategy-based-on-resistance-support-levels-volume-and-10-wma-in-fa3eb250241


# # Connect to the Interactive Brokers API
# ib = IB()
# ib.connect()
# Define the stock and the period
# stock = yf.Ticker("TSLA")
# period = '1d'
# Download the historical data
def download_historical(ticker, start_date, end_date=None):
    # data = stock.history(period=period)
    data = download_ohlc(ticker, start_date, end_date)

    # Compute the resistance and support levels
    data['resistance'] = data['High'].rolling(window=20).max()
    data['support'] = data['Low'].rolling(window=20).min()
    # Compute the 10-day WMA
    data['wma'] = data['Close'].rolling(window=10).mean()

    # Create a new stock object for the stock
    # stock_obj = Stock(stock.symbol, 'SMART', 'USD')
    return data


def run_strategy(volume_threshold, ticker, start_date, end_date=None):
    # Define the trading strategy
    while True:
        # Get the latest data
        # data = stock.history(period=period)
        data = download_historical(ticker, start_date, end_date)
        data['resistance'] = data['High'].rolling(window=20).max()
        data['support'] = data['Low'].rolling(window=20).min()
        data['wma'] = data['Close'].rolling(window=10).mean()

        # Check if the current price breaks above the resistance level and volume is greater than a threshold value
        if (data['Close'].iloc[-1] > data['resistance'].iloc[-1]) and (data['Volume'].iloc[-1] > volume_threshold) and (
                data['Close'].iloc[-1] > data['wma'].iloc[-1]):
            # Place a buy order
            # ib.qualifyContracts(stock_obj)
            # order = ib.placeOrder(stock_obj, 'BUY', 1)
            print("BUY ORDER")
        # Check if the current price breaks below the support level and volume is greater than a threshold value
        elif (data['Close'].iloc[-1] < data['support'].iloc[-1]) and (data['Volume'].iloc[-1] > volume_threshold) and (
                    data['Close'].iloc[-1] < data['wma'].iloc[-1]):
            # Place a sell order
            # ib.qualifyContracts(stock_obj)
            # order = ib.placeOrder(stock_obj, 'SELL', 1)
            print("SELL ORDER")

