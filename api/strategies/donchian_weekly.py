import pandas as pd
import requests
import pandas_ta as ta
import matplotlib.pyplot as plt
from termcolor import colored as cl
import math

plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')

# https://levelup.gitconnected.com/an-algo-trading-strategy-which-made-8-371-a-python-case-study-58ed12a492dc


# EXTRACTING HISTORICAL DATA
def get_historical_data(symbol, start_date, interval):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {"token": "YOUR API KEY", "symbols": f"{symbol}", "from": f"{start_date}", "interval": f"{interval}"}
    hist_json = requests.get(url, params=querystring).json()
    df = pd.DataFrame(hist_json[0]['candles'])
    return df


def calc_donchian_channel(df):
    # CALCULATING DONCHIAN CHANNEL
    df[['dcl', 'dcm', 'dcu']] = df.ta.donchian(lower_length=40, upper_length=50)
    df = df.dropna().drop('time', axis=1).rename(columns={'dateTime': 'date'})
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df.tail()
    return df


def plot_donchian_channel(ticker, df):
    # PLOTTING DONCHIAN CHANNEL
    plt.plot(df[-300:].close, label='CLOSE')
    plt.plot(df[-300:].dcl, color='black', linestyle='--', alpha=0.3)
    plt.plot(df[-300:].dcm, color='orange', label='DCM')
    plt.plot(df[-300:].dcu, color='black', linestyle='--', alpha=0.3, label='DCU,DCL')
    plt.legend()
    plt.title(f'{ticker} DONCHIAN CHANNELS 50')
    plt.xlabel('Date')
    plt.ylabel('Close')


# BACKTESTING THE STRATEGY
def implement_strategy(df, investment):
    in_position = False
    equity = investment

    for i in range(3, len(df)):
        if df['high'][i] == df['dcu'][i] and in_position == False:
            no_of_shares = math.floor(equity / df.close[i])
            equity -= (no_of_shares * df.close[i])
            in_position = True
            print(cl('BUY: ', color='green', attrs=['bold']),
                  f'{no_of_shares} Shares are bought at ${df.close[i]} on {str(df.index[i])[:10]}')
        elif df['low'][i] == df['dcl'][i] and in_position == True:
            equity += (no_of_shares * df.close[i])
            in_position = False
            print(cl('SELL: ', color='red', attrs=['bold']),
                  f'{no_of_shares} Shares are bought at ${df.close[i]} on {str(df.index[i])[:10]}')
    if in_position == True:
        equity += (no_of_shares * df.close[i])
        print(cl(f'\nClosing position at {df.close[i]} on {str(df.index[i])[:10]}', attrs=['bold']))
        in_position = False

    earning = round(equity - investment, 2)
    roi = round(earning / investment * 100, 2)
    print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs=['bold']))


def calc_spy_buy_hold():
    spy = get_historical_data('SPY', '1993-01-01', '1W')
    spy_ret = round(((spy.close.iloc[-1] - spy.close.iloc[0]) / spy.close.iloc[0]) * 100)
    print(cl('SPY ETF buy/hold return:', attrs=['bold']), f'{spy_ret}%')


def run_donchian_weekly(ticker, start_date):
    df = get_historical_data(ticker, start_date, '1W')
    df.tail()
    df = calc_donchian_channel(df)
    plot_donchian_channel(ticker, df)
    implement_strategy(df, 100000)
    calc_spy_buy_hold()


if __name__ == "__main__":
    run_donchian_weekly('AAPL', '1993-01-01')








