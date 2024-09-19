import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

# https://medium.datadriveninvestor.com/downloading-a-large-amount-of-historical-stock-data-using-the-financial-modeling-prep-api-540128ae4929
# https://utm.guru/ugBtL


api_key = 'your api'
base_url = 'https://financialmodelingprep.com/api/v3/historical-chart/'


def get_jsonparsed_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data from {url}")
        return None


def fetch_batch(symbol, interval, start, end, batch_size):
    data_frames = []

    date_ranges = pd.date_range(start=start, end=end, freq=f'{batch_size}D')

    for i in range(len(date_ranges) - 1):
        batch_start = date_ranges[i].strftime('%Y-%m-%d')

        if i > 0:
            batch_start = (date_ranges[i] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        batch_end = date_ranges[i + 1].strftime('%Y-%m-%d')

        url = f'{base_url}{interval}/{symbol}?from={batch_start}&to={batch_end}&apikey={api_key}'
        data = get_jsonparsed_data(url)

        if data is not None:
            df = pd.DataFrame(data)
            data_frames.append(df)

    if data_frames:
        result_df = pd.concat(data_frames, ignore_index=True)

        return result_df
    else:
        return None


def basic_analysis(data):
    basic_stats = data['close'].describe()
    return basic_stats


def returns_analysis(data):
    data['returns'] = data['close'].pct_change()
    returns_stats = data['returns'].describe()
    return returns_stats


def plot_historical(ticker, data):
    data_sorted = data.sort_values(by='date')

    sampled_data = data_sorted.sample(n=2000)

    sampled_data['date'] = pd.to_datetime(sampled_data['date'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='close', data=sampled_data, label='Close Price')
    plt.title(f'{ticker} Stock Price Over Time (Sampled Data)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45, ha='right')

def plot_trading_volume(ticker, data):
    data_sorted = data.sort_values(by='date')

    sampled_data = data_sorted.sample(n=1000)

    sampled_data['date'] = pd.to_datetime(sampled_data['date'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='volume', data=sampled_data, label='Volume', color='grey')
    plt.title(f'{ticker} Trading Volume Over Time (Sampled Data)')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ticker = 'SPY'

    batch_size = 2
    data = fetch_batch(ticker, '1min', '2003-01-01', '2023-12-19', batch_size)

    ba = basic_analysis(data)
    ra = returns_analysis(data)
    print(ba)
    print(ra)

    plot_historical(ticker, data)
    plot_trading_volume(ticker, data)







