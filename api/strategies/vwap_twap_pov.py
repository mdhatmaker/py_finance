import config as cfg
from eodhd import APIClient

# https://eodhd.medium.com/advanced-trading-strategies-maximizing-profits-with-vwap-twap-and-pov-using-python-987e0ead97f1


api = APIClient(cfg.API_KEY)


def get_ohlc_data():
    df = api.get_historical_data("GSPC.INDX", "d", results=730)
    return df


# Volume Weighted Average Price
def calc_vwap(df):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    print(df[['vwap']])
    return df


# Time Weighted Average Price
def calc_twap(df):
    df['average_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['twap'] = df['average_price'].expanding().mean()
    print(df[['twap']])
    return df


# Percent of Volume
def calc_pov(df, order_size, pov_rate=0.20):
    # order_size = 800  # Total shares to be executed
    # pov_rate = 0.20  # 20% of market volume
    df['daily_execution_target'] = df['volume'] * pov_rate
    df['actual_execution'] = df['daily_execution_target'].apply(lambda x: min(x, order_size))
    order_size -= df['actual_execution'].sum()
    print(df[['volume', 'daily_execution_target', 'actual_execution']])
    return df


if __name__ == "__main__":
    df = get_ohlc_data()
    print(df)







"""
Trading Signals Using VWAP

A buy signal occurs when a stock’s price is below the VWAP, indicating it is undervalued (similar to
the Relative Strength Index, RSI). The rationale is that the stock is trading at a price lower than
its average, suggesting a potential upward movement towards the average.

On the other hand, a sell signal is generated when the price is above the VWAP, implying the stock is
overvalued. This suggests a possible downward correction to bring the price in line with the average.

Trading Signals with TWAP

A buy signal could be generated when the current price is below the TWAP, indicating that the price
 is lower than the average, suggesting a potential upward correction.
 
Conversely, a sell signal might be considered when the current price is above the TWAP, indicating
that the price is higher than the average, suggesting a potential downward correction.

How PoV Works

PoV is defined as the ratio of the trader’s order size to the total market volume over a specified
time frame, expressed as a percentage. For instance, if a trader sets a PoV rate of 20%, the trader
intends to execute their order such that it constitutes no more than 20% of the market’s total volume
in each time slice, thus reducing market impact.

"""