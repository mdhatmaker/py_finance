import yfinance as yf
import pandas as pd
from nixtla import NixtlaClient
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
from config import NIXTLA_API_TOKEN

# https://medium.com/modern-ai/i-just-tried-timegpt-the-new-chatgpt-like-model-for-forecasting-these-are-the-results-3e6872641501


nixtla_client = NixtlaClient(api_key=NIXTLA_API_TOKEN)
nixtla_client.validate_api_key()


def download_df_dailychange_volume(ticker_symbol: str, startDate: str = '2020-01-01', endDate: str = None) -> DataFrame:
    if not endDate:
        endDate = datetime.now().strftime('%Y-%m-%d')
    # Download data for a single symbol, e.g., "^GSPC" *S&P 500 index)
    print(f'{ticker_symbol}    {startDate} to {endDate}')
    data = yf.download(ticker_symbol, start=startDate, end=endDate)
    df = pd.DataFrame(data)
    # Calculate the daily change from open to close
    df['Daily Change'] = 100*(df['Close'] - df['Open']) / df['Open']
    df['Volume'] = df['Volume'] / 1000000
    # Create a timestamp column and change the index
    df['timestamp'] = df.index
    df = df.reset_index(drop=True)
    return df


"""
The model has a very easy to use API, you just need to give the model the following:
    df=         the dataframe you are working with
    h=          the time horizon you want to predict (5 days ahead in my case)
    freq=       the time unit (’D’, for days in my case=
    time_col=   the column with time stamps (’timestamp’)
    target_col= the column you want to predict (’Close’ in my case)
"""

def generate_forecasts(df):
    # Create a list with intervals to iterate
    my_list=np.arange(0,250,5)
    # Store real price difference for weeks in this list
    reals = []
    # Store weekly forecasted price differences in this list
    validates = []
    count = 0
    for i in my_list:
        count += 1
        if count > 23:
            pass
        ndf = pd.DataFrame()
        ndf = df[:1000+i]
        now = df.iloc[1000+i]['Close']
        validate = df.iloc[1000+i+5]['Close']
        reals.append(validate-now)
        forecast = nixtla_client.forecast(df=ndf, h=5, freq='D', time_col='timestamp', target_col='Close')
        validates.append(forecast.iloc[4]['TimeGPT']-forecast.iloc[0]['TimeGPT'])

    # Compare the signs of elements in both lists
    count = sum((r == 0 and v == 0) or (r > 0 and v > 0) or (r < 0 and v < 0) for r, v in zip(reals, validates))
    print(count)


def run_timegpt_forecast(ticker: str, start_date: str, end_date: str):
    #ticker = "^GSPC"
    df = download_df_dailychange_volume(ticker, start_date, end_date)
    generate_forecasts(df)


