import yfinance as yf
import re
import pandas as pd
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# https://medium.datadriveninvestor.com/exploring-unusual-options-activity-with-python-and-apis-bd93ba25f3c2


INTRINIO_API_KEY = 'OjdiN2E3ZGVlZTIxYTNjOGZmNWQ1NjEzNzU2OTMyNjU0'

sns.set()


def get_options_for_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        all_data = []
        for expiration_date in options_dates:
            options = stock.option_chain(expiration_date)
            calls = options.calls
            puts = options.puts
            for _, call in calls.iterrows():
                all_data.append({'Symbol': call['contractSymbol']})
            for _, put in puts.iterrows():
                all_data.append({'Symbol': put['contractSymbol']})
        return pd.DataFrame(all_data)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def get_historical_price_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="max")
        data.index = data.index.strftime('%Y-%m-%d')
        return data
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def Extract_Expiration_Date(text):
    pattern = r'^[A-Z]+(\d{6})[A-Z]+\d{5}\d+'
    match = re.search(pattern, text)
    if match:
        captured_digits = match.group(1)
        transformed_string = '20' + captured_digits[:2] + '-' + captured_digits[2:4] + '-' + captured_digits[4:]
        return pd.to_datetime(transformed_string)
    else:
        return None


def extract_strike(option_string):
    # Use regular expression to extract the last 8 digits
    match = re.search(r'\d{8}$', option_string)
    if match:
        number_str = match.group()
        # Insert decimal and remove leading zeroes
        number_str = number_str[:-3].lstrip('0') + '.' + number_str[-3:]
        return float(number_str)
    else:
        return None


def historical_eod_options_data(contract_symbol, api_key):
    try:
        url = f'https://api-v2.intrinio.com/options/prices/{contract_symbol}/eod?api_key={api_key}'
        eod_options_json = requests.get(url).json()
        df = pd.DataFrame(eod_options_json['prices'])
        df_sorted = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        df_sorted = df_sorted[['date', 'open_interest', 'volume']]

        ## Add expire date
        expire_date = Extract_Expiration_Date(contract_symbol)
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        df_sorted['DTE'] = (expire_date - df_sorted['date']).dt.days

        ## Strike Price
        strike = extract_strike(contract_symbol)

        ## Option Type
        match = re.search(r'.{8}(.).{8}$', contract_symbol)
        option_type = match.group(1)

        ## Add New Columns
        df_sorted['Strike'] = strike
        df_sorted['Option_Type'] = option_type
        df_sorted['DTE'] = (expire_date - df_sorted['date']).dt.days

        return df_sorted
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Contract format incorrect or does not exist on this date")
        return None


def get_option_data(ohlc_prices, options_symbol_data):
    hist_options_dfs = []

    for i in tqdm(range(len(options_symbol_data))):
      s = options_symbol_data['Symbol'][i]
      hist_data = historical_eod_options_data(contract_symbol = s,api_key = INTRINIO_API_KEY)
      hist_options_dfs.append(hist_data)\

    hist_option_df = pd.concat(hist_options_dfs).reset_index(drop=True)

    hist_option_df['date'] = pd.to_datetime(hist_option_df['date'])
    ohlc_prices.index = pd.to_datetime(ohlc_prices.index)
    ohlc_prices = ohlc_prices[['Close']]
    merged_df = pd.merge(hist_option_df, ohlc_prices, left_on='date', right_index=True, how='left')

    merged_df['Moneyness'] = None

    for i in tqdm(range(len(merged_df))):
        option_type = merged_df['Option_Type'][i]
        strike = merged_df['Strike'][i]
        close = merged_df['Close'][i]
        if (option_type == 'C') & (strike > close):
            merged_df['Moneyness'][i] = 'otm'
        elif (option_type == 'P') & (strike > close):
            merged_df['Moneyness'][i] = 'itm'
        elif (option_type == 'C') & (strike < close):
            merged_df['Moneyness'][i] = 'itm'
        else:
            merged_df['Moneyness'][i] = 'otm'

    return merged_df


def reformat_data(merged_df):
    # Filter observations where 'Moneyness' is 'itm' and 'Option_Type' is 'C' (in-the-money calls)
    itm_calls_df = merged_df[(merged_df['Moneyness'] == 'itm') & (merged_df['Option_Type'] == 'C')]

    # Filter observations where 'Moneyness' is 'itm' and 'Option_Type' is 'P' (in-the-money puts)
    itm_puts_df = merged_df[(merged_df['Moneyness'] == 'itm') & (merged_df['Option_Type'] == 'P')]

    # Filter observations where 'Moneyness' is 'otm' and 'Option_Type' is 'C' (out-of-the-money calls)
    otm_calls_df = merged_df[(merged_df['Moneyness'] == 'otm') & (merged_df['Option_Type'] == 'C')]

    # Filter observations where 'Moneyness' is 'otm' and 'Option_Type' is 'P' (out-of-the-money puts)
    otm_puts_df = merged_df[(merged_df['Moneyness'] == 'otm') & (merged_df['Option_Type'] == 'P')]

    # Group by 'date' and calculate total open interest and volume, and count of observations for in-the-money calls
    grouped_itm_calls = itm_calls_df.groupby('date').agg({'open_interest': 'sum', 'volume': 'sum', 'date': 'count'})
    grouped_itm_calls = grouped_itm_calls.rename(columns={'date': 'count'})
    grouped_itm_calls['avg_itm_call_oi_per_contract'] = grouped_itm_calls['open_interest'] / grouped_itm_calls['count']
    grouped_itm_calls['avg_itm_call_volume_per_contract'] = grouped_itm_calls['volume'] / grouped_itm_calls['count']
    grouped_itm_calls = grouped_itm_calls.reset_index()

    # Group by 'date' and calculate total open interest and volume, and count of observations for in-the-money puts
    grouped_itm_puts = itm_puts_df.groupby('date').agg({'open_interest': 'sum', 'volume': 'sum', 'date': 'count'})
    grouped_itm_puts = grouped_itm_puts.rename(columns={'date': 'count'})
    grouped_itm_puts['avg_itm_put_oi_per_contract'] = grouped_itm_puts['open_interest'] / grouped_itm_puts['count']
    grouped_itm_puts['avg_itm_put_volume_per_contract'] = grouped_itm_puts['volume'] / grouped_itm_puts['count']
    grouped_itm_puts = grouped_itm_puts.reset_index()

    # Group by 'date' and calculate total open interest and volume, and count of observations for out-of-the-money calls
    grouped_otm_calls = otm_calls_df.groupby('date').agg({'open_interest': 'sum', 'volume': 'sum', 'date': 'count'})
    grouped_otm_calls = grouped_otm_calls.rename(columns={'date': 'count'})
    grouped_otm_calls['avg_otm_call_oi_per_contract'] = grouped_otm_calls['open_interest'] / grouped_otm_calls['count']
    grouped_otm_calls['avg_otm_call_volume_per_contract'] = grouped_otm_calls['volume'] / grouped_otm_calls['count']
    grouped_otm_calls = grouped_otm_calls.reset_index()

    # Group by 'date' and calculate total open interest and volume, and count of observations for out-of-the-money puts
    grouped_otm_puts = otm_puts_df.groupby('date').agg({'open_interest': 'sum', 'volume': 'sum', 'date': 'count'})
    grouped_otm_puts = grouped_otm_puts.rename(columns={'date': 'count'})
    grouped_otm_puts['avg_otm_put_oi_per_contract'] = grouped_otm_puts['open_interest'] / grouped_otm_puts['count']
    grouped_otm_puts['avg_otm_put_volume_per_contract'] = grouped_otm_puts['volume'] / grouped_otm_puts['count']
    grouped_otm_puts = grouped_otm_puts.reset_index()

    # Merge all DataFrames on 'date'
    merge_one = pd.merge(grouped_itm_calls, grouped_itm_puts, on='date', how='outer')
    merge_one = merge_one[['date','avg_itm_call_oi_per_contract', 'avg_itm_call_volume_per_contract','avg_itm_put_oi_per_contract','avg_itm_put_volume_per_contract']]
    merged_two = pd.merge(merge_one, grouped_otm_calls, on='date', how='outer')
    merged_two = merged_two[['date', 'avg_itm_call_oi_per_contract', 'avg_itm_call_volume_per_contract','avg_itm_put_oi_per_contract','avg_itm_put_volume_per_contract','avg_otm_call_oi_per_contract', 'avg_otm_call_volume_per_contract']]
    merged_three = pd.merge(merged_two, grouped_otm_puts, on='date', how='outer')
    merged_three = merged_three[['date', 'avg_itm_call_oi_per_contract',
           'avg_itm_call_volume_per_contract', 'avg_itm_put_oi_per_contract',
           'avg_itm_put_volume_per_contract', 'avg_otm_call_oi_per_contract',
           'avg_otm_call_volume_per_contract','avg_otm_put_oi_per_contract', 'avg_otm_put_volume_per_contract']]

    # Calculate median value in 'Close' column for each date
    median_close = merged_df.groupby('date')['Close'].median().reset_index()
    final_df = pd.merge(merged_three, median_close, on='date', how='left')
    final_df = final_df.sort_values(by='date').dropna().reset_index(drop=True)
    final_df['next_day_return'] = final_df['Close'].pct_change().shift(-1)

    return final_df


def calculate_option_changes(final_df):
    contract_analytics_columns = ['avg_itm_call_oi_per_contract',
           'avg_itm_call_volume_per_contract', 'avg_itm_put_oi_per_contract',
           'avg_itm_put_volume_per_contract', 'avg_otm_call_oi_per_contract',
           'avg_otm_call_volume_per_contract', 'avg_otm_put_oi_per_contract',
           'avg_otm_put_volume_per_contract']

    delta_df = final_df[contract_analytics_columns].pct_change()
    delta_columns = [f'{col}_change' for col in contract_analytics_columns]

    # Rename columns to represent day-over-day changes
    delta_df.columns = delta_columns

    # Concatenate changes_df with the original DataFrame df
    final_df_with_deltas = pd.concat([final_df, delta_df], axis=1).dropna()
    # delta_df = final_df[contract_analytics_columns].pct_change()
    # delta_columns = [f'{col}_change' for col in contract_analytics_columns]

    return final_df_with_deltas, delta_columns


def plot_unusual_options_activity(final_df_with_deltas, delta_columns):
    Next_Day_Returns = []

    for c in delta_columns:
        sorted_df = final_df_with_deltas.sort_values(by=c, ascending=False)
        sorted_df = sorted_df[['date', c, 'next_day_return']]
        Next_Day_Returns.append(sorted_df['next_day_return'].head(1).values[0])

    # Example lists of strings and values
    delta_columns_shortened = ['OI_itm_calls', 'Volume_itm_calls', 'OI_itm_puts', 'Volume_itm_puts', 'OI_otm_calls', 'Volume_otm_calls',
                               'OI_otm_puts', 'Volume_otm_puts']
    labels = delta_columns_shortened
    values = Next_Day_Returns

    # Set the style
    sns.set_style("darkgrid")

    # Set the size of the plot
    plt.figure(figsize=(20, 12))

    # Create bar plot using seaborn
    sns.barplot(x=labels, y=values)

    # Add horizontal line with value 5
    avg_return = sorted_df['next_day_return'].median()
    plt.axhline(y=avg_return, color='r', linestyle='--')

    # Rotate x-axis labels to 45 degrees and increase font size
    plt.xticks(rotation=45, fontsize=12)

    # Increase font size of y-axis labels
    plt.yticks(fontsize=14)

    # Convert y-axis tick labels to percentage format
    plt.gca().set_yticklabels(['{:.3f}%'.format(val) for val in plt.gca().get_yticks()])

    # Show the plot
    plt.show()


def run_unusual_options_activity(ticker):
    options_symbol_data = get_options_for_ticker(ticker)
    ohlc_prices = get_historical_price_data(ticker)
    merged_df = get_option_data(ohlc_prices, options_symbol_data)
    final_df = reformat_data(merged_df)
    final_df_with_deltas, delta_columns = calculate_option_changes(final_df)
    plot_unusual_options_activity(final_df_with_deltas, delta_columns)


if __name__ == "__main__":
    run_unusual_options_activity('AAPL')

