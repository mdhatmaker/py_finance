import yfinance as yf
import talib
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import matplotlib.gridspec as gridspec
from pandas import Series
from datetime import datetime
import time

# https://medium.com/@crisvelasquez/automating-61-candlestick-trading-patterns-in-python-f28709c50684


pattern_funcs = [
    ("Two Crows", talib.CDL2CROWS, "bearish"),
    ("Three Black Crows", talib.CDL3BLACKCROWS, "bearish"),
    ("Three Inside Up/Down", talib.CDL3INSIDE, "bearish"),
    ("Three-Line Strike", talib.CDL3LINESTRIKE, "bearish"),
    ("Three Outside Up/Down", talib.CDL3OUTSIDE, "bearish"),
    ("Three Stars In The South", talib.CDL3STARSINSOUTH, "bearish"),
    ("Three Advancing White Soldiers", talib.CDL3WHITESOLDIERS, "bearish"),
    ("Abandoned Baby", talib.CDLABANDONEDBABY, "bearish"),
    ("Advance Block", talib.CDLADVANCEBLOCK, "bearish"),
    ("Belt-hold", talib.CDLBELTHOLD, "bearish"),
    ("Breakaway", talib.CDLBREAKAWAY, "bearish"),
    ("Closing Marubozu", talib.CDLCLOSINGMARUBOZU, "bearish"),
    ("Concealing Baby Swallow", talib.CDLCONCEALBABYSWALL, "bearish"),
    ("Counterattack", talib.CDLCOUNTERATTACK, "bearish"),
    ("Dark Cloud Cover", talib.CDLDARKCLOUDCOVER, "bearish"),
    ("Doji", talib.CDLDOJI, "bearish"),
    ("Doji Star", talib.CDLDOJISTAR, "bearish"),
    ("Dragonfly Doji", talib.CDLDRAGONFLYDOJI, "bearish"),
    ("Engulfing Pattern", talib.CDLENGULFING, "bearish"),
    ("Evening Doji Star", talib.CDLEVENINGDOJISTAR, "bearish"),
    ("Evening Star", talib.CDLEVENINGSTAR, "bearish"),
    ("Up/Down-gap side-by-side white lines", talib.CDLGAPSIDESIDEWHITE, "bearish"),
    ("Gravestone Doji", talib.CDLGRAVESTONEDOJI, "bearish"),
    ("Hammer", talib.CDLHAMMER, "bearish"),
    ("Hanging Man", talib.CDLHANGINGMAN, "bearish"),
    ("Harami Pattern", talib.CDLHARAMI, "bearish"),
    ("Harami Cross Pattern", talib.CDLHARAMICROSS, "bearish"),
    ("High-Wave Candle", talib.CDLHIGHWAVE, "bearish"),
    ("Hikkake Pattern", talib.CDLHIKKAKE, "bearish"),
    ("Modified Hikkake Pattern", talib.CDLHIKKAKEMOD, "bearish"),
    ("Homing Pigeon", talib.CDLHOMINGPIGEON, "bearish"),
    ("Identical Three Crows", talib.CDLIDENTICAL3CROWS, "bearish"),
    ("In-Neck Pattern", talib.CDLINNECK, "bearish"),
    ("Inverted Hammer", talib.CDLINVERTEDHAMMER, "bearish"),
    ("Kicking", talib.CDLKICKING, "bearish"),
    ("Kicking - bull/bear determined by the longer marubozu", talib.CDLKICKINGBYLENGTH, "bearish"),
    ("Ladder Bottom", talib.CDLLADDERBOTTOM, "bearish"),
    ("Long Legged Doji", talib.CDLLONGLEGGEDDOJI, "bearish"),
    ("Long Line Candle", talib.CDLLONGLINE, "bearish"),
    ("Marubozu", talib.CDLMARUBOZU, "bearish"),
    ("Matching Low", talib.CDLMATCHINGLOW, "bearish"),
    ("Mat Hold", talib.CDLMATHOLD, "bearish"),
    ("Morning Doji Star", talib.CDLMORNINGDOJISTAR, "bearish"),
    ("Morning Star", talib.CDLMORNINGSTAR, "bearish"),
    ("On-Neck Pattern", talib.CDLONNECK, "bearish"),
    ("Piercing Pattern", talib.CDLPIERCING, "bearish"),
    ("Rickshaw Man", talib.CDLRICKSHAWMAN, "bearish"),
    ("Rising/Falling Three Methods", talib.CDLRISEFALL3METHODS, "bearish"),
    ("Separating Lines", talib.CDLSEPARATINGLINES, "bearish"),
    ("Shooting Star", talib.CDLSHOOTINGSTAR, "bearish"),
    ("Short Line Candle", talib.CDLSHORTLINE, "bearish"),
    ("Spinning Top", talib.CDLSPINNINGTOP, "bearish"),
    ("Stalled Pattern", talib.CDLSTALLEDPATTERN, "bearish"),
    ("Stick Sandwich", talib.CDLSTICKSANDWICH, "bearish"),
    ("Takuri (Dragonfly Doji with very long lower shadow)", talib.CDLTAKURI, "bearish"),
    ("Tasuki Gap", talib.CDLTASUKIGAP, "bearish"),
    ("Thrusting Pattern", talib.CDLTHRUSTING, "bearish"),
    ("Tristar Pattern", talib.CDLTRISTAR, "bearish"),
    ("Unique 3 River", talib.CDLUNIQUE3RIVER, "bearish"),
    ("Upside Gap Two Crows", talib.CDLUPSIDEGAP2CROWS, "bearish"),
    ("Upside/Downside Gap Three Methods", talib.CDLXSIDEGAP3METHODS, "bearish")
]


def download_historical_data(symbol: str, startDate: str = '2020-01-01', endDate: str = None) -> Series:
    #symbol = "SAP.DE"
    if not endDate:
        endDate = datetime.now()
    print(f'{symbol}    {startDate} to {endDate}')
    stock_data = yf.download(symbol, start=startDate, end=endDate)
    # close_prices = stock_data['Close']
    # return close_prices
    return stock_data


def identify_patterns(symbol, data):
    for pattern_name, pattern_func, bull_bear in pattern_funcs:
        data[pattern_name] = pattern_func(data['Open'], data['High'], data['Low'], data['Close'])
        pattern_dates = data[data[pattern_name] != 0].index

        # Skip if there are no detected patterns of this type
        if len(pattern_dates) != 0:

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 4)

            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            custom_style = mpf.make_mpf_style(marketcolors=mc)

            ax1 = plt.subplot(gs[0, :3])
            data[['Close']].plot(ax=ax1, color='blue')
            for date in pattern_dates:
                ax1.axvline(date, color='red', linestyle='--', label=pattern_name if pattern_name not in [l.get_label() for l in ax1.lines] else "")
                ax1.annotate(date.strftime('%Y-%m-%d'), (date, data['Close'].loc[date]), xytext=(-15, 10 + 20),
                             textcoords='offset points', color='red', fontsize=12, rotation=90)

            window = 5  # Days before and after the pattern
            for i in range(5):
                if len(pattern_dates) > i:
                    pattern_date = pattern_dates[-(i + 1)]

                    start_date = pattern_date - pd.Timedelta(days=window)
                    end_date = min(data.index[-1], pattern_date + pd.Timedelta(days=window))
                    valid_dates = pd.date_range(start=start_date, end=end_date).intersection(data.index)

                    subset = data.loc[valid_dates]

                    if i == 0:
                        ax = plt.subplot(gs[0, 3])
                    else:
                        ax = plt.subplot(gs[1, i - 1])

                    mpf.plot(subset, type='candle', ax=ax, volume=False, show_nontrading=False, style=custom_style)
                    ax.set_title(f'{pattern_name} Pattern {i + 1} for {symbol}')

                    x_ticks = list(range(0, len(valid_dates), 1))
                    x_labels = [date.strftime('%Y-%m-%d') for date in valid_dates]
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=90)

            ax1.set_title(f"{symbol} Stock Price and {pattern_name} Pattern Detection")
            ax1.legend(loc='best')
            ax1.grid(True)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
            plt.tight_layout()
            plt.show()

        time.sleep(2.5)


def run_candlestick_ta_patterns(symbol: str):
    symbol='SAP.DE'
    data = download_historical_data(symbol)
    identify_patterns(symbol, data)

