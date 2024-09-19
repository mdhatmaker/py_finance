from datetime import datetime, timedelta
from api.indicators.volatility_indicators.average_true_range import run_average_true_range, calculate_atr
from api.utils.yahoo_finance import download_ohlc


# Use ATR to calculate Exit (Profit) and Stop (Loss) order levels:
#
# - Use ATR for order calculations (i.e. if ATR is 0.0055 then ATR_pips is 55 pips).
#
# - For a simple 2.00 risk-reward ratio (risking half of what we expect to gain)...
#
## 1. use ATR to enter Buy orders
##
## Buy at current market price.
## Take profit at current market price + (2 x ATR_pips).
## Stop the position at current market price — (1 x ATR_pips).
##
## 2. use ATR to enter Sell orders
##
## Sell at current market price.
## Take profit at current market price - (2 x ATR_pips).
## Stop the position at current market price + (1 x ATR_pips).


BUY: int = 1
SELL: int = -1


def round_str(f: float):
    return "{:.2f}".format(f)


def calc_stop_exit_offsets_atr(ticker, start_date=None, end_date=None, risk=1.0, reward=2.0, window=14):
    if not start_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * window)
        end_date = end_date.strftime("%Y-%m-%d")
        start_date = start_date.strftime("%Y-%m-%d")
    data = download_ohlc(ticker, start_date, end_date)
    print(data.tail)
    atr = calculate_atr(data, window=window)
    most_recent_atr = atr[-1]
    profit_offset = reward * most_recent_atr
    stop_offset = risk * most_recent_atr
    print(f'{ticker}   ATR:{round_str(most_recent_atr)}   stop_offset:{round_str(stop_offset)}   profit_offset:{round_str(profit_offset)}')
    return stop_offset, profit_offset, most_recent_atr


def calc_stop_exit_levels_atr(buy_sell, current_price, profit_offset, stop_offset):
    exit_price = current_price + buy_sell * profit_offset
    stop_price = current_price - buy_sell * stop_offset
    return round(stop_price, 2), round(exit_price, 2)


def calc_stop_exit_levels_buy(current_price, profit_offset, stop_offset):
    stop_, exit_ = calc_stop_exit_levels_atr(BUY, current_price, profit_offset, stop_offset)
    return stop_, exit_


def calc_stop_exit_levels_sell(current_price, profit_offset, stop_offset):
    stop_, exit_ = calc_stop_exit_levels_atr(SELL, current_price, profit_offset, stop_offset)
    return stop_, exit_


def calc_stop_exit_buy(current_price, ticker, start_date=None, end_date=None, risk=1.0, reward=2.0, window=14):
    stop_offset, profit_offset, atr = calc_stop_exit_offsets_atr(ticker, start_date, end_date, risk, reward, window)
    stop_price, exit_price = calc_stop_exit_levels_buy(current_price, profit_offset, stop_offset)
    return stop_price, exit_price


def calc_stop_exit_sell(current_price, ticker, start_date=None, end_date=None, risk=1.0, reward=2.0, window=14):
    stop_offset, profit_offset, atr = calc_stop_exit_offsets_atr(ticker, start_date, end_date, risk, reward, window)
    stop_price, exit_price = calc_stop_exit_levels_sell(current_price, profit_offset, stop_offset)
    return stop_price, exit_price


def calc_stop_exit_atr(current_price, ticker, start_date=None, end_date=None, risk = 1.0, reward=2.0, window=14):
    stop_buy, exit_buy = calc_stop_exit_buy(current_price, ticker, start_date, end_date, risk, reward, window)
    stop_sell, exit_sell = calc_stop_exit_sell(current_price, ticker, start_date, end_date, risk, reward, window)
    result = {'stop_buy': stop_buy, 'exit_buy': exit_buy, 'stop_sell': stop_sell, 'exit_sell:': exit_sell}
    return result


def run_stop_exit_atr(current_price, ticker, start_date=None, end_date=None, risk = 1.0, reward=2.0, window=14):
    result = calc_stop_exit_atr(current_price, ticker, start_date, end_date, risk, reward, window)
    print(current_price, result)
    return result


if __name__ == "__main__":
    ticker = 'AAPL'
    current_market_price = 214.28
    stop_exit = run_stop_exit_atr(current_market_price, ticker)
    stop_exit = run_stop_exit_atr(current_market_price, ticker, None, None, 1.5, 3.0)

