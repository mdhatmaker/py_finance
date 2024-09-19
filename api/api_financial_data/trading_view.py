# from tvDatafeed import TvDatafeed, Interval
# from tvd import Segment
import tvd
from tvd import Segment
#from Segment import



tv = TvDatafeed()


# Specify the ticker and the market from which you want to pull data:
TICKER = 'SPX'
MARKET = 'CBOE'

# Fetch the last 5000 bars of 1-minute data for the ticker:
df = tv.get_hist(symbol=TICKER, exchange=MARKET, interval=Interval.in_1_minute, n_bars=5000, extended_session=False)

# Before saving, format the DataFrame to ensure it contains only essential information and is reader-friendly:
df = df.reset_index()
df = df[['datetime','open','high','low','close','volume']]
df.columns = ['Time','Open','High','Low','Last','Volume']
df = df.sort_values(by=['Time'])
df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.strftime('%m/%d/%Y %H:%M')

# Generate a filename based on the ticker and the date range of the data, then save the DataFrame as a CSV file:
min_date = df['Time'].min().strftime('%Y-%m-%d')
max_date = df['Time'].max().strftime('%Y-%m-%d')
filename = f"{TICKER.lower()}_{min_date}_{max_date}.csv"
df.to_csv(filename, index=False)
