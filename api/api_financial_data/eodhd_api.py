import sys
import pandas as pd
from eodhd import APIClient
from datetime import datetime

# https://eodhd.medium.com/backtesting-trading-strategies-with-python-pandas-28db61a2b791


EODHD_API_KEY = "<Your_EODHD_API_Key>"


def get_ohlc_data(ticker, start_date, end_date=None) -> pd.DataFrame:
    """Return a DataFrame of OHLC data"""
    # ticker = "HSPX.LSE"
    # start_date = "2020-01-01"
    # end_date = "2023-09-08"
    if not end_date:
        end_date = datetime.now().strftime('Y%-m%-d%')
    api = APIClient(EODHD_API_KEY)
    df = api.get_historical_data(symbol=ticker, interval="d", iso8601_start=start_date, iso8601_end=end_date)
    df.drop(columns=["symbol", "interval", "close"], inplace=True)
    df.rename(columns={"adjusted_close": "close"}, inplace=True)
    print(df)

    return pd.DataFrame([], columns=["date", "open", "high", "low", "close"])


def simulate_trades(df):
    balance_base = 0
    account_balance = 1000
    buy_order_quote = 1000
    is_order_open = False
    orders = []
    sell_value = 0

    for index, row in df.iterrows():
        if row["buy_signal"] and is_order_open == 0:
            is_order_open = 1

            if sell_value < 1000 and sell_value > 0:
                buy_order_quote = sell_value
            else:
                buy_order_quote = 1000

            buy_amount = buy_order_quote / row["close"]
            balance_base += buy_amount
            account_balance += sell_value

            order = {
                "timestamp": index,
                "account_balance": account_balance,
                "buy_order_quote": buy_order_quote,
                "buy_order_base": buy_amount
            }

            account_balance -= buy_order_quote

        if row["sell_signal"] and is_order_open == 1:
            is_order_open = 0

            sell_value = buy_amount * row["close"]
            balance_base -= buy_amount

            order["sell_order_quote"] = sell_value
            order["profit"] = order["sell_order_quote"] - order["buy_order_quote"]
            order["margin"] = (order["profit"] / order["buy_order_quote"]) * 100

            orders.append(order)
        print(index)

    df_orders = pd.DataFrame(orders)
    print(df_orders)
    return df_orders


def backtest_ema12_ema26(df):
    """Backtest a strategy using pandas"""

    # # ticker = "HSPX.LSE"
    # df = get_ohlc_data(ticker, "2020-01-01", None)

    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    df["ema12gtema26"] = df["ema12"] > df["ema26"]
    df["buy_signal"] = df["ema12gtema26"].ne(df["ema12gtema26"].shift())
    df.loc[df["ema12gtema26"] == False, "buy_signal"] = False  # noqa: E712
    df["buy_signal"] = df["buy_signal"].astype(int)

    df["ema12ltema26"] = df["ema12"] < df["ema26"]
    df["sell_signal"] = df["ema12ltema26"].ne(df["ema12ltema26"].shift())
    df.loc[df["ema12ltema26"] == False, "sell_signal"] = False  # noqa: E712
    df["sell_signal"] = df["sell_signal"].astype(int)

    df.drop(columns=["ema12", "ema26", "ema12gtema26", "ema12ltema26"], inplace=True)

    print(df)
    return df



def main() -> int:
    """Backtest a strategy using pandas"""

    df = get_ohlc_data("HSPX.LSE", "2020-01-01", None)
    print(df)

    df = backtest_ema12_ema26(df)
    simulate_trades(df)

    return 0


if __name__ == '__main__':
    sys.exit(main())

