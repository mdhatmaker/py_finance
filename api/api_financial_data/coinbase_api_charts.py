import requests
import ast
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Best Stock Market APIs — Worldwide Markets & Exchanges
# https://medium.com/coinmonks/best-stock-market-apis-ae1efb739ac4

# curl "https://api.exchange.coinbase.com/products/btc-usd/candles?granularity=86400&start=2023-05-01T13:39:38.120597&end=2024-02-25T13:39:38.120597"

# Coinbase API docs
# https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
"""
product: currency pair like btc-usd, eth-usd, btc-eur
granularity: must be one of the following “minute” values - {60, 300, 900, 3600, 21600, 86400}
start: start timestamp in ISO format
end: end timestamp in ISO format
"""


def get_hist_prices(coin, base_currency, days):
    """
    Get historical prices for a given coin
    :param coin: e.g. btc, eth, dot
    :param base_currency: fiat currency like usd, eur, gbp
    :param days: number of days to go back
    :return:
    """
    now = datetime.now()
    then = now - timedelta(days=days)
    then, now = then.isoformat(), now.isoformat()
    url = f"https://api.exchange.coinbase.com/products/{coin}-{base_currency}/candles?granularity=86400&start={then}" \
          f"&end={now}"
    headers = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=headers)
    res = response.text
    res_list = ast.literal_eval(res)
    cols = ['timestamp', 'price_low', 'price_high', 'price_open', 'price_close']
    data_all = []
    for data in res_list:
        data_all.append(dict(zip(cols, data)))
    hist_df = pd.DataFrame(data_all)
    hist_df['timestamp'] = hist_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    hist_df['timestamp'] = hist_df['timestamp'].dt.normalize()
    hist_df.index = hist_df['timestamp']
    hist_df = hist_df[hist_df.columns[~hist_df.columns.isin(['timestamp'])]]
    hist_df = hist_df.sort_index()
    return hist_df


def bollinger_bands(df, window=20):
    df['sma'] = df['price_close'].rolling(window).mean()
    df['std_dev'] = df['price_close'].rolling(window).std()
    df['upper_band'] = df['sma'] + 2 * df['std_dev']
    df['lower_band'] = df['sma'] - 2 * df['std_dev']
    return df


def plot_charts(coins):
    """
    Accepts a list of coins and plots a chart for each coin
    For now, it only accepts a list size of up to 6, but this can of course be extended to accept more by making the
    subplot size dynamic based on the number of coins in our argument
    :param coins:
    :return:
    """
    plt.style.use('dark_background')
    figure = plt.figure(figsize=(25, 12))
    # Create a 3x2 grid of subplots
    subplot_pos = 320
    current_date = datetime.now()
    first_day_of_year = datetime(current_date.year, 1, 1)
    for idx, coin in enumerate(coins):
        df = get_hist_prices(coin, 'usd', 300)
        df = bollinger_bands(df)
        # determine position of the current chart within our subplot grid
        pos = subplot_pos + idx + 1
        ax = figure.add_subplot(pos)
        ax.yaxis.tick_right()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        ax.set_title(coin.upper(), fontweight='bold', loc='left', color='orange')

        # determine the price change and percentage change from the previous day for our annotation
        start_price, last_price = df['price_close'].iloc[-2], df['price_close'].iloc[-1]
        price_change = last_price - start_price
        pct_change = last_price / start_price - 1
        color = 'lime' if price_change > 0 else 'red'
        decimal_count = str(last_price)[::-1].find('.')
        sign = '+' if price_change > 0 else ''
        annotation = f"{last_price} {sign}{price_change:.{decimal_count}f} " \
                     f"{sign}{pct_change:.2%}"
        ax.annotate(annotation, xy=(0, 1), xycoords='axes fraction', fontsize=12,
                    xytext=(5, -5), textcoords='offset points', ha='left', va='top', color=color)

        # determine first date of the year and calculate YTD performance. If the first date of the year is not present,
        # we reindex the dataframe and forward fill the missing values
        if first_day_of_year in df.index:
            ytd_start = df.loc[first_day_of_year, 'price_close']
        else:
            df = df.reindex(df.index.append(pd.DatetimeIndex([first_day_of_year]))).sort_index().ffill()
            ytd_start = df.loc[first_day_of_year, 'price_close']
        ytd_pct = last_price / ytd_start - 1
        ytd_color = 'lime' if ytd_pct > 0 else 'red'
        ax.annotate(f"YTD: {ytd_pct:.2%}", xy=(0, 1), xycoords='axes fraction', fontsize=12,
                    xytext=(5, -20), textcoords='offset points', ha='left', va='top', color=ytd_color)

        # I prefer my charts without borders, so spines are removed and ticks are turned off
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        # this is where we fill the area between the price and the lower band to give it that Bloomberg feel
        ax.fill_between(df.index, df['price_close'], df['lower_band'].dropna().min(), color='blue', alpha=0.3)

        # finally we plot the line charts for the price, upper and lower bands
        ax.plot(df['price_close'], color='white')
        ax.plot(df['upper_band'], color='lime')
        ax.plot(df['lower_band'], color='orange')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.show()


if __name__ == '__main__':
    coins = ['btc', 'eth', 'dot', 'link', 'sol', 'ada']
    plot_charts(coins)


