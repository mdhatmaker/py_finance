# Import libraries
import pandas as pd
import yfinance as yf
import datetime as dt
from pandas_datareader import DataReader
#from fbprophet import Prophet
import matplotlib.pyplot as plt

# https://medium.com/@serdarilarslan/predicting-stock-prices-with-prophet-a-python-guide-7b773d821fef


def get_historical_data(ticker: str, num_years=5):
    # ticker = 'AAPL'
    # num_years = 20
    start_date = dt.datetime.now() - dt.timedelta(days=365.25 * num_years)
    end_date = dt.datetime.now()

    # Fetch stock data
    #data = DataReader(ticker, 'yahoo', start_date, end_date)
    data = yf.download(ticker, start=start_date, end=end_date)

    # Prepare data for Prophet
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    return data


def generate_forecast(data):
    # Create and fit Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    # Create future dataframe and make predictions
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return model, forecast


def plot_forecast(ticker, model, forecast):
    # Plot predictions
    model.plot(forecast)
    plt.title(f"Predicted Stock Price of {ticker} using Prophet")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()


def run_prophet_forecast(ticker: str, num_years=20):
    #ticker = 'AAPL'
    data = get_historical_data(ticker, num_years)
    model, forecast = generate_forecast(data)
    plot_forecast(ticker, model, forecast)


