import os
from datetime import datetime
from pandas import DataFrame, Series
import yfinance as yf
import pandas as pd


def get_income_statement(ticker):
    # Fetch income statement data from Yahoo Finance
    stock = yf.Ticker(ticker)
    income_statement = stock.financials.loc['Net Income']
    return income_statement


def download_ohlc(symbol: str, startDate: str = '2018-01-01', endDate: str = None) -> DataFrame:
    if not endDate:
        endDate = datetime.now().strftime('%Y-%m-%d')
    # Download data for a single ticker symbol, e.g., AAPL
    print(f'{symbol}    {startDate} to {endDate}')
    # df = yf.download('AAPL', start='2020-05-22', end='2024-05-22')
    df = yf.download(symbol, start=startDate, end=endDate)
    return df


def download_close_prices(symbol: str, startDate: str = '2018-01-01', endDate: str = None) -> Series:
    if not endDate:
        endDate = datetime.now().strftime('%Y-%m-%d')
    # Download data for a single ticker symbol, e.g., AAPL
    print(f'{symbol}    {startDate} to {endDate}')
    data = download_ohlc(symbol, startDate, endDate)
    close_prices = data['Close']
    return close_prices


def download_returns(symbol: str, startDate: str = '2018-01-01', endDate: str = None) -> Series:
    price_data = download_close_prices(symbol, startDate, endDate)
    price_data_pct_change = price_data.pct_change().dropna()
    return price_data_pct_change


def load_ticker_ts_df(ticker: str, start_date: str, end_date: str = None):
    """
    Load and cache time series financial data from Yahoo Finance API.
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    - start_date (str): The start date in 'YYYY-MM-DD' format for data retrieval.
    - end_date (str): The end date in 'YYYY-MM-DD' format for data retrieval.
    Returns:
    - df (pandas.DataFrame): A DataFrame containing the financial time series data."""
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    # data_dir_path = os.getenv('DATA_DIR_PATH', default='./files/data')
    data_dir_path = '/Users/michael/git/nashed/t1/webTraderApp/files/data'
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

