from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://medium.com/@crisvelasquez/acquiring-and-analyzing-earnings-announcements-data-in-python-fbd610cebc19


def fetch_earnings_data(ticker):
    # Set up Selenium to run headlessly
    options = Options()
    options.headless = True
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")

    driver = webdriver.Chrome(options=options)
    url = f"https://finance.yahoo.com/calendar/earnings?symbol={ticker}"
    driver.get(url)

    # Find the rows of the earnings table
    rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')

    data = []

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, 'td')
        cols = [elem.text for elem in cols]
        data.append(cols)

    # Close the WebDriver
    driver.quit()

    # Assuming the data structure is as expected, create a DataFrame
    columns = ['Symbol', 'Company', 'Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)']
    df = pd.DataFrame(data, columns=columns)

    return df


def fetch_clean_earnings_data(ticker):
    #ticker = "SAP"
    earnings_data = fetch_earnings_data(ticker)

    # Extract the time and timezone information into a new column
    earnings_data['Earnings Time'] = earnings_data['Earnings Date'].str.extract(r'(\d{1,2} [AP]MEDT)')

    # Extract just the date part from the "Earnings Date" column
    earnings_data['Earnings Date'] = earnings_data['Earnings Date'].str.extract(r'(\b\w+ \d{1,2}, \d{4})')

    # Convert string date to datetime
    earnings_data['Earnings Date'] = pd.to_datetime(earnings_data['Earnings Date'], format='%b %d, %Y')

    # Convert datetime to desired string format
    earnings_data['Earnings Date'] = earnings_data['Earnings Date'].dt.strftime('%Y-%m-%d')

    earnings_data['Surprise(%)'] = earnings_data['Surprise(%)'].str.replace('+', '').astype(float)

    return earnings_data


# earnings_data['Surprise(%)'] = earnings_data['Surprise(%)'].str.replace('+', '').astype(float)
# earnings_data['Earnings Date'] = pd.to_datetime(earnings_data['Earnings Date'])


def plot_earnings_prices(stock_data, earnings_data):
    # Plotting stock data
    plt.figure(figsize=(25, 7))
    stock_data['Close'].plot(label='Stock Price', color='blue')

    # Plotting earnings surprise
    for index, row in earnings_data.iterrows():
        date = row['Earnings Date']
        # If exact date is not available, use the closest available date
        if date not in stock_data.index:
            date = stock_data.index[stock_data.index.get_loc(date, method='nearest')]

        if row['Surprise(%)'] > 0:
            color = 'green'
            marker = '^'
        else:
            color = 'red'
            marker = 'v'

        plt.plot(date, stock_data.loc[date, 'Close'], marker, color=color, markersize=15)

    plt.title(f'{ticker} Stock Price with Earnings Surprise', fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def earnings_nearby_prices(ticker_symbol, earnings_data):
    # Assuming 'earnings_data' is the DataFrame and has an 'Earnings Date' column in string format
    earnings_data['Earnings Date'] = pd.to_datetime(earnings_data['Earnings Date'])

    # Now add buffer days to the start and end dates
    buffer_days = 10
    startDate = earnings_data['Earnings Date'].min() - pd.Timedelta(days=buffer_days)
    endDate = earnings_data['Earnings Date'].max() + pd.Timedelta(days=buffer_days)

    stock_data = yf.download(ticker_symbol, start=startDate, end=endDate)
    return stock_data


# Function to compute price effect
def compute_price_effect(earnings_date, stock_data):
    try:
        # For "Price Before", if missing, we use the most recent previous price
        price_before = stock_data.loc[:pd.Timestamp(earnings_date) - pd.Timedelta(days=1), 'Close'].ffill().iloc[-1]

        price_on = stock_data.loc[pd.Timestamp(earnings_date), 'Close']

        # For "Price After", if missing, we use the next available price
        price_after = stock_data.loc[pd.Timestamp(earnings_date) + pd.Timedelta(days=1):, 'Close'].bfill().iloc[0]

        price_effect = ((price_after - price_before) / price_before) * 100
    except (KeyError, IndexError):  # in case the date is missing in the stock_data even after filling
        return None, None, None, None
    return price_before, price_on, price_after, price_effect


#df = pd.DataFrame(data)
#df['Earnings Date'] = pd.to_datetime(df['Earnings Date'])


def plot_earnings_effects(earnings_data):
    # Sort the dataframe by 'Earnings Date' in ascending order
    latest_earnings_data = earnings_data.sort_values(by='Earnings Date').tail(14)

    # Setting up the plot
    fig, ax1 = plt.subplots(figsize=(30,8))

    # Bar positions
    positions = range(len(latest_earnings_data ))
    width = 0.25
    r1 = [pos - width for pos in positions]
    r2 = positions
    r3 = [pos + width for pos in positions]

    # Clustered bar plots for prices
    bars1 = ax1.bar(r1, latest_earnings_data ['Price Before'], width=width, label='Price Before', color='blue', edgecolor='grey')
    bars2 = ax1.bar(r2, latest_earnings_data ['Price On'], width=width, label='Price On', color='cyan', edgecolor='grey')
    bars3 = ax1.bar(r3, latest_earnings_data ['Price After'], width=width, label='Price After', color='lightblue', edgecolor='grey')

    # Line plots for Surprise(%) and Price Effect (%)
    ax2 = ax1.twinx()
    ax2.plot(positions, latest_earnings_data ['Surprise(%)'], color='red', marker='o', label='Surprise(%)')
    ax2.plot(positions, latest_earnings_data ['Price Effect (%)'], color='green', marker='o', label='Price Effect (%)')

    # Annotations for the Surprise(%) and Price Effect (%)
    for i, (date, surprise, effect) in enumerate(zip(latest_earnings_data ['Earnings Date'], latest_earnings_data ['Surprise(%)'], latest_earnings_data ['Price Effect (%)'])):
        ax2.annotate(f"{surprise}%", (i, surprise), textcoords="offset points", xytext=(0,10), ha='center', fontsize=16, color='red', fontweight='bold')
        ax2.annotate(f"{effect:.2f}%", (i, effect), textcoords="offset points", xytext=(0,10), ha='center', fontsize=16, color='green', fontweight='bold')

    # Annotations for prices
    def annotate_bars(bars, ax):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=14, rotation=45)

    annotate_bars(bars1, ax1)
    annotate_bars(bars2, ax1)
    annotate_bars(bars3, ax1)

    # Setting x-axis with better spacing
    ax1.set_xticks(positions)
    ax1.set_xticklabels(latest_earnings_data ['Earnings Date'].dt.strftime('%Y-%m-%d'), rotation=45, ha='right', fontsize=14)

    # Setting labels and title
    ax1.set_xlabel('Earnings Date', fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('Earnings Data with Surprise and Price Effect', fontsize=18)

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_earnings_surprises(earnings_data):
    # Drop rows with NaN values in 'Surprise(%)' and 'Price Effect (%)' columns
    filtered_earnings_data = earnings_data.dropna(subset=['Surprise(%)', 'Price Effect (%)'])

    # Linear regression
    slope, intercept = np.polyfit(filtered_earnings_data['Surprise(%)'], filtered_earnings_data['Price Effect (%)'], 1)
    x = np.array(filtered_earnings_data['Surprise(%)'])
    y_pred = slope * x + intercept

    # Compute r-squared
    correlation_matrix = np.corrcoef(filtered_earnings_data['Surprise(%)'], filtered_earnings_data['Price Effect (%)'])
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    # Scatter plot with regression line
    plt.figure(figsize=(30, 8))
    plt.scatter(filtered_earnings_data['Surprise(%)'], filtered_earnings_data['Price Effect (%)'], color='blue', marker='o')
    plt.plot(x, y_pred, color='red', label=f'y={slope:.3f}x + {intercept:.3f}')  # regression line
    plt.title('Earnings Surprise vs. Price Effect', fontsize=20)
    plt.xlabel('Earnings Surprise(%)')
    plt.ylabel('Price Effect(%)')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.annotate(f'R-squared = {r_squared:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15, color='green')
    plt.show()



# # Define the ticker symbol
# tickerSymbol = 'SAP'
#
# # Check the minimum and maximum earnings dates from the 'df' DataFrame
# min_earnings_date = earnings_data['Earnings Date'].min()
# max_earnings_date = earnings_data['Earnings Date'].max()
#
# # Get data on this ticker
# stock_data = yf.Ticker(tickerSymbol)
#
# # Get the historical prices for this ticker within the range of earnings dates
# hist = stock_data.history(start=min_earnings_date, end=max_earnings_date)
#
# # Make the datetime index timezone-naive for compatibility with earnings dates
# hist.index = hist.index.tz_localize(None)
#
# # Initialize an empty list to hold the price series
# price_series_list = []
#
# # Extract relevant price data
# for index, row in earnings_data[['Earnings Date']].iterrows():
#     earnings_date = pd.to_datetime(row['Earnings Date']).date()
#
#     # Adjust the start date to ensure there's data available for forward-filling
#     extended_start_date = earnings_date - timedelta(days=7)  # extending to ensure we have data to forward-fill
#     start_date = earnings_date - timedelta(days=5)
#     end_date = earnings_date + timedelta(days=5)
#
#     # Select the stock prices for the extended date range
#     prices = hist.loc[extended_start_date:end_date, 'Close']
#
#     if prices.empty:
#         print(f"No price data available for the range {extended_start_date} to {end_date}. Skipping.")
#         continue
#
#     # Forward-fill missing values, this time with available data due to extended range
#     all_days = pd.date_range(start=extended_start_date, end=end_date, freq='D')
#     prices = prices.reindex(all_days, method='ffill')
#
#     # Truncate the prices Series to only the date range we're interested in (i.e., -5 to +5 days around earnings)
#     prices = prices.loc[start_date:end_date]
#
#     # Normalize prices based on the closing price 5 days before earnings
#     prices /= prices.iloc[0]
#
#     # Add the series to the list with the days relative to earnings as the new index
#     price_series_list.append(prices.reset_index(drop=True))  # reset_index for proper alignment during concatenation
#
# # Check if the price_series_list is empty
# if not price_series_list:
#     raise ValueError("No price data was added to the list. Please check your input data and date ranges.")
#
# # Concatenate all the series into a single DataFrame
# price_data = pd.concat(price_series_list, axis=1)
#
# # Correcting the index to represent days relative to earnings
# price_data.index = np.arange(-5, 6)
#
# # Now, let's plot each series correctly
# plt.figure(figsize=(25, 10))
#
# # Iterate over each series and plot
# for column in price_data.columns:
#     plt.plot(price_data.index, price_data[column])  # Each series represents a different earnings date
#
# plt.axvline(x=0, color='red', linestyle='--', label='Earnings Date', linewidth=5)
# plt.xticks(np.arange(-5, 6, 1))  # Ensuring the x-axis reflects -5 to +5 days
#
# # Adding title and labels
# plt.title('SAP Stock Prices Around Earnings Announcements', fontsize=15)
# plt.xlabel('Days Relative to Earnings', fontsize=12)
# plt.ylabel('Normalized Price', fontsize=12)
#
# # Set the tick size
# plt.tick_params(axis='both', which='major', labelsize=12)  # Increase tick label size
#
# #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')  # Adjusted the legend position so it doesn't overlap the plot
# plt.grid(True)
# plt.show()






if __name__ == "__main__":

    ticker = 'SAP'

    earnings_data = fetch_clean_earnings_data(ticker)
    print(earnings_data)


    # Fetch stock price data
    stock_data = yf.download(ticker, start=earnings_data['Earnings Date'].min(), end=earnings_data['Earnings Date'].max())


    # Price effect of earnings
    earnings_data['Price Before'], earnings_data['Price On'], earnings_data['Price After'], earnings_data['Price Effect (%)'] = (
        zip(*earnings_data['Earnings Date'].apply(compute_price_effect, stock_data=stock_data)))
    # earnings_data['Surprise(%)'] = earnings_data['Surprise(%)'].str.replace('+', '').astype(float)
    print(earnings_data)


    plot_earnings_surprises(earnings_data)





