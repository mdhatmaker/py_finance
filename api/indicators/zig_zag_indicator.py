from datetime import datetime, timedelta
import yfinance as yf
import mplfinance as mpf

# https://medium.com/@larry.prestosa/zigzag-indicator-for-stock-trading-39450b3d03c7


def calculate_zigzag(data, percentage=5):
    """
    Calculate the Zigzag indicator.

    :param data: A Pandas DataFrame with a 'Close' column.
    :param percentage: The minimum percentage change to consider.
    :return: A Pandas DataFrame with a 'Zigzag' column.
    """
    last_peak = last_trough = data['Close'][0]
    direction = 0  # 1 for upward trend, -1 for downward trend
    zigzag = [last_peak]

    for close_price in data['Close'][1:]:
        if direction >= 0:  # currently in an upward trend or no trend
            change = (close_price / last_trough - 1) * 100
            if change > percentage:
                zigzag.extend([None] * (len(zigzag) - 1) + [last_trough, close_price])
                direction = 1
                last_peak = close_price
            elif close_price < last_trough:
                last_trough = close_price
        else:  # currently in a downward trend
            change = (last_peak / close_price - 1) * 100
            if change > percentage:
                zigzag.extend([None] * (len(zigzag) - 1) + [last_peak, close_price])
                direction = -1
                last_trough = close_price
            elif close_price > last_peak:
                last_peak = close_price

    return zigzag


def plot_zigzag(ticker, df):
    # Add Zigzag as an additional plot
    apd = mpf.make_addplot(df['Zigzag'], type='line', color='blue')

    # Customize plot style
    mpf_style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 8})

    # Plot it
    mpf.plot(df, style=mpf_style, type='hollow_and_filled', addplot=apd,
             volume=True, figscale=1.5,
             title=f'{ticker} Stock Price - Zigzag Indicator',
             ylabel='Price ($)',
             xrotation=20,
             datetime_format='%b-%d',
             tight_layout=True)


def run_zig_zag(ticker: str,  start_date: str, end_date: str, percentage=5):
    # Download price data
    df = yf.download(ticker, start=start_date, end=end_date)
    # Calculate indicator and added to the dataframe
    df['Zigzag'] = calculate_zigzag(df, percentage)
    # Plot
    plot_zigzag(ticker, df)


if __name__ == "__main__":

    ticker = 'TSLA'

    # Download price data for the past 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)

    run_zig_zag(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

