from itertools import combinations
from statsmodels.tsa.stattools import coint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf
import numpy as np
import os


# https://medium.datadriveninvestor.com/citadels-strategy-anyone-can-use-pairs-trading-7b81428a6c67


crypto_forex_stocks = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'ETC-USD', 'XLM-USD', 'AAVE-USD', 'EOS-USD', 'XTZ-USD', 'ALGO-USD', 'XMR-USD', 'KCS-USD',
                       'MKR-USD', 'BSV-USD', 'RUNE-USD', 'DASH-USD', 'KAVA-USD', 'ICX-USD', 'LINA-USD', 'WAXP-USD', 'LSK-USD', 'EWT-USD', 'XCN-USD', 'HIVE-USD', 'FTX-USD', 'RVN-USD', 'SXP-USD', 'BTCB-USD']
bank_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'DB', 'UBS', 'BBVA', 'SAN', 'ING', ' BNPQY', 'HSBC', 'SMFG', 'PNC', 'USB', 'BK', 'STT', 'KEY', 'RF', 'HBAN', 'FITB',  'CFG',
               'BLK', 'ALLY', 'MTB', 'NBHC', 'ZION', 'FFIN', 'FHN', 'UBSI', 'WAL', 'PACW', 'SBCF', 'TCBI', 'BOKF', 'PFG', 'GBCI', 'TFC', 'CFR', 'UMBF', 'SPFI', 'FULT', 'ONB', 'INDB', 'IBOC', 'HOMB']
global_indexes = ['^DJI', '^IXIC', '^GSPC', '^FTSE', '^N225', '^HSI', '^AXJO', '^KS11', '^BFX', '^N100',
                  '^RUT', '^VIX', '^TNX']


def load_ticker_ts_df(ticker, start_date, end_date):
    """
    Load and cache time series financial data from Yahoo Finance API.
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    - start_date (str): The start date in 'YYYY-MM-DD' format for data retrieval.
    - end_date (str): The end date in 'YYYY-MM-DD' format for data retrieval.
    Returns:
    - df (pandas.DataFrame): A DataFrame containing the financial time series data."""
    data_dir_path = os.getenv('DATA_DIR_PATH')   #'/Users/michael/data'
    dir_path = f'{data_dir_path}/cached'
    cached_file_path = f'{dir_path}/{ticker}_{start_date}_{end_date}.pkl'
    try:
        if os.path.exists(cached_file_path):
            df = pd.read_pickle(cached_file_path)
        else:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            df.to_pickle(cached_file_path)
    except FileNotFoundError:
        print(f'Error downloading and caching or loading file with ticker: {ticker}')

    return df


def sanitize_data(data_map, start_date, end_date):
    TS_DAYS_LENGTH = (pd.to_datetime(end_date) -
                      pd.to_datetime(start_date)).days
    data_sanitized = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for ticker, data in data_map.items():
        if data is None or len(data) < (TS_DAYS_LENGTH / 2):
            # We cannot handle shorter TSs
            continue
        if len(data) > TS_DAYS_LENGTH:
            # Normalize to have the same length (TS_DAYS_LENGTH)
            data = data[-TS_DAYS_LENGTH:]
        # Reindex the time series to match the date range and fill in any blanks (Not Numbers)
        data = data.reindex(date_range)
        data['Adj Close'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Adj Close'].interpolate(method='linear', inplace=True)
        data['Adj Close'].fillna(method='pad', inplace=True)
        data['Adj Close'].fillna(method='bfill', inplace=True)
        assert not np.any(np.isnan(data['Adj Close'])) and not np.any(
            np.isinf(data['Adj Close']))
        data_sanitized[ticker] = data
    return data_sanitized


def find_cointegrated_pairs(tickers_ts_map, p_value_threshold=0.2):
    """
    Find cointegrated pairs of stocks based on the Augmented Dickey-Fuller (ADF) test.
    Parameters:
    - tickers_ts_map (dict): A dictionary where keys are stock tickers and values are time series data.
    - p_value_threshold (float): The significance level for cointegration testing.
    Returns:
    - pvalue_matrix (numpy.ndarray): A matrix of cointegration p-values between stock pairs.
    - pairs (list): A list of tuples representing cointegrated stock pairs and their p-values.
    """
    tickers = list(tickers_ts_map.keys())
    n = len(tickers)
    # Extract 'Adj Close' prices into a matrix (each column is a time series)
    adj_close_data = np.column_stack(
        [tickers_ts_map[ticker]['Adj Close'].values for ticker in tickers])
    pvalue_matrix = np.ones((n, n))
    # Calculate cointegration p-values for unique pair combinations
    for i, j in combinations(range(n), 2):
        result = coint(adj_close_data[:, i], adj_close_data[:, j])
        pvalue_matrix[i, j] = result[1]
    pairs = [(tickers[i], tickers[j], pvalue_matrix[i, j])
             for i, j in zip(*np.where(pvalue_matrix < p_value_threshold))]
    return pvalue_matrix, pairs


def plot_heatmap(uts_sanitized, pvalues, P_VALUE_THRESHOLD = 0.02):
    plt.figure(figsize=(26, 26))
    heatmap = sns.heatmap(pvalues, xticklabels=uts_sanitized.keys(),
                          yticklabels=uts_sanitized.keys(), cmap='RdYlGn_r',
                          mask=(pvalues > (P_VALUE_THRESHOLD)),
                          linecolor='gray', linewidths=0.5)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), size=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), size=14)
    plt.show()


def get_sorted_pairs(pairs):
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=False)
    sorted_pairs = sorted_pairs[0:35]
    sorted_pairs_labels, pairs_p_values = zip(*[(f'{y1} <-> {y2}', p * 1000) for y1, y2, p in sorted_pairs])
    return sorted_pairs_labels, pairs_p_values


def plot_sorted_pvalues(sorted_pairs_labels, pairs_p_values):
    plt.figure(figsize=(12, 18))
    plt.barh(sorted_pairs_labels,
             pairs_p_values, color='red')
    plt.xlabel('P-Values (1000)', fontsize=8)
    plt.ylabel('Pairs', fontsize=6)
    plt.title('Cointegration P-Values (in 1000s)', fontsize=20)
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.show()


def plot_pairs_time_series(ticker_pairs, uts_sanitized):
    # ticker_pairs = [("AAVE-USD", "C"), ("XMR-USD", "C"), ("FTX-USD", "ALLY")]
    fig, axs = plt.subplots(3, 1, figsize=(18, 14))
    scaler = MinMaxScaler()
    for i, (ticker1, ticker2) in enumerate(ticker_pairs):
        # Scale the price data for each pair using MIN MAX
        scaled_data1 = scaler.fit_transform(
            uts_sanitized[ticker1]['Adj Close'].values.reshape(-1, 1))
        scaled_data2 = scaler.fit_transform(
            uts_sanitized[ticker2]['Adj Close'].values.reshape(-1, 1))
        axs[i].plot(scaled_data1, label=f'{ticker1}', color='lightgray', alpha=0.7)
        axs[i].plot(scaled_data2, label=f'{ticker2}', color='lightgray', alpha=0.7)
        # Apply rolling mean with a window of 15
        scaled_data1_smooth = pd.Series(scaled_data1.flatten()).rolling(
            window=15, min_periods=1).mean()
        scaled_data2_smooth = pd.Series(scaled_data2.flatten()).rolling(
            window=15, min_periods=1).mean()
        axs[i].plot(scaled_data1_smooth, label=f'{ticker1} SMA', color='red')
        axs[i].plot(scaled_data2_smooth, label=f'{ticker2} SMA', color='blue')
        axs[i].set_ylabel('*Scaled* Price $', fontsize=12)
        axs[i].set_title(f'{ticker1} vs {ticker2}', fontsize=18)
        axs[i].legend()
        axs[i].set_xticks([])
    plt.tight_layout()
    plt.show()


def calculate_zscore(ticker_pair, uts_sanitized):
    ticker1, ticker2 = ticker_pair
    # ticker1 = "AAVE-USD"
    # ticker2 = "C"
    TRAIN = int(len(uts_sanitized[ticker1]) * 0.85)
    TEST = len(uts_sanitized[ticker1]) - TRAIN

    ticker1_ts = uts_sanitized[ticker1]["Adj Close"][:TRAIN]
    ticker2_ts = uts_sanitized[ticker2]["Adj Close"][:TRAIN]
    # Calculate price ratio (ticker price / C price)
    ratios = ticker2_ts / ticker1_ts

    return ratios


def plot_zscore(ticker_pair, ratios):
    ticker1, ticker2 = ticker_pair
    fig, ax = plt.subplots(figsize=(12, 8))
    ratios_mean = np.mean(ratios)
    ratios_std = np.std(ratios)
    ratios_zscore = (ratios - ratios_mean) / ratios_std
    ax.plot(ratios.index, ratios_zscore, label="Z-Score", color='blue')
    # Plot reference lines
    ax.axhline(1.0, color="green", linestyle='--', label="Upper Threshold (1.0)")
    ax.axhline(-1.0, color="red", linestyle='--', label="Lower Threshold (-1.0)")
    ax.axhline(0, color="black", linestyle='--', label="Mean")
    ax.set_title(f'{ticker1} / {ticker2}: Price Ratio and Z-Score', fontsize=18)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price Ratio / Z-Score')
    ax.legend()
    plt.tight_layout()
    plt.show()

def signals_zscore_evolution(ticker1_ts, ticker2_ts, window_size=15, first_ticker=True):
    """
    Generate trading signals based on z-score analysis of the ratio between two time series.
    Parameters:
    - ticker1_ts (pandas.Series): Time series data for the first security.
    - ticker2_ts (pandas.Series): Time series data for the second security.
    - window_size (int): The window size for calculating z-scores and ratios' statistics.
    - first_ticker (bool): Set to True to use the first ticker as the primary signal source, and False to use the second.Returns:
    - signals_df (pandas.DataFrame): A DataFrame with 'signal' and 'orders' columns containing buy (1) and sell (-1) signals.
    """
    ratios = ticker1_ts / ticker2_ts
    ratios_mean = ratios.rolling(
        window=window_size, min_periods=1, center=False).mean()
    ratios_std = ratios.rolling(
        window=window_size, min_periods=1, center=False).std()
    z_scores = (ratios - ratios_mean) / ratios_std
    buy = ratios.copy()
    sell = ratios.copy()
    if first_ticker:
        # These are empty zones, where there should be no signal
        # the rest is signalled by the ratio.
        buy[z_scores > -1] = 0
        sell[z_scores < 1] = 0
    else:
        buy[z_scores < 1] = 0
        sell[z_scores > -1] = 0
    signals_df = pd.DataFrame(index=ticker1_ts.index)
    signals_df['signal'] = np.where(buy > 0, 1, np.where(sell < 0, -1, 0))
    signals_df['orders'] = signals_df['signal'].diff()
    signals_df.loc[signals_df['orders'] == 0, 'orders'] = None
    return signals_df


def plot_buy_sell(ticker_pair, uts_sanitized):  #, signals_df1, signals_df2):
    ticker1, ticker2 = ticker_pair
    ts1 = uts_sanitized[ticker1]["Adj Close"]
    ts2 = uts_sanitized[ticker2]["Adj Close"]
    plt.figure(figsize=(26, 18))
    signals_df1 = signals_zscore_evolution(ts1, ts2)
    profit_df1 = calculate_profit(signals_df1, ts1)
    ax1, _ = plot_strategy(ts1, signals_df1, profit_df1)
    signals_df2 = signals_zscore_evolution(ts1, ts2, first_ticker=False)
    profit_df2 = calculate_profit(signals_df2, ts2)
    ax2, _ = plot_strategy(ts2, signals_df2, profit_df2)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_title(f'{ticker2} Paired with {ticker1}', fontsize=18)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_title(f'{ticker1} Paired with {ticker2}', fontsize=18)
    plt.tight_layout()
    plt.show()


# Plot sum of returns from ALL signals
def plot_returns(ticker_pair, profit_df1, profit_df2):
    ticker1, ticker2 = ticker_pair
    plt.figure(figsize=(12, 6))
    cumulative_profit_combined = profit_df1 + profit_df2
    ax2_combined = cumulative_profit_combined.plot(label='Profit%', color='green')
    plt.legend(loc='upper left', fontsize=10)
    plt.title(f'{ticker1} & {ticker2} Paired - Cumulative Profit', fontsize=18)
    plt.tight_layout()
    plt.show()


def run_citadel_pairs(start_date: str, end_date: str, P_VALUE_THRESHOLD = 0.02):

    universe_tickers = crypto_forex_stocks + bank_stocks + global_indexes
    universe_tickers_ts_map = {ticker: load_ticker_ts_df(ticker, start_date, end_date) for ticker in universe_tickers}

    # Sample some
    uts_sanitized = sanitize_data(universe_tickers_ts_map, start_date, end_date)
    shapes = uts_sanitized['JPM'].shape, uts_sanitized['BTC-USD'].shape

    # This section can take up to 5 mins
    # P_VALUE_THRESHOLD = 0.02
    pvalues, pairs = find_cointegrated_pairs(uts_sanitized, p_value_threshold=P_VALUE_THRESHOLD)

    # Plot Cointegration P-Values (sorted)
    sorted_pairs_labels, pairs_p_values = get_sorted_pairs(pairs)
    plot_sorted_pvalues(sorted_pairs_labels, pairs_p_values)

    plot_heatmap(uts_sanitized, pvalues)

    ticker_pairs = [("AAVE-USD", "C"), ("XMR-USD", "C"), ("FTX-USD", "ALLY")]
    plot_pairs_time_series(ticker_pairs, uts_sanitized)

    ticker_pair = ("AAVE-USD", "C")
    ratios = calculate_zscore(ticker_pair, uts_sanitized)
    plot_zscore(ticker_pair, ratios)

    # plot_returns(ticker_pair, profit_df1, profit_df2)


if __name__ == "__main__":

    run_citadel_pairs(start_date='2021-01-01', end_date='2023-10-31')




"""
To create trading signals, we’ll utilize the Z-score and mean with a rolling window approach, eliminating the
need to split the data into training and test sets. The Z-score is calculated as follows:

The Z-score measures the standardization of a price series relative to its historical mean using a rolling window approach.
It is calculated as follows:

Z = (X - u) / sigma

Where:
-X is the price we want to standardize.
-u is the mean (average) of the rolling window.
-sigma is the standard deviation of the rolling window.

The Z-score quantifies how far the current ratio of the two asset prices is from its historical mean.
When the Z-score exceeds a predefined threshold, typically +1 or -1, it generates a trading signal.

A Z-score above +1 indicates that one asset is overvalued relative to the other, signaling a sell for the
overvalued asset and a buy for the undervalued one.

Conversely, if the Z-score falls below -1, it suggests the undervalued asset has become overvalued, prompting a
sell for the former and a buy for the latter.

This strategy exploits mean-reversion principles to capitalize on temporary divergences and the expectation of a return to the mean.
"""

