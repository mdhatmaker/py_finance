import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from typing import List, Dict
from sklearn.model_selection import GridSearchCV


# XGBoost forecasting docs
# https://medium.com/@bugragultekin/xgboost-for-stock-price-forecasting-64f89719a8e4
"""
"""


def download_close_prices(symbol: str, startDate: str = '2020-05-22', endDate: str = '2024-05-22') -> Series:
    # Download data for a single stock, e.g., AAPL
    print(f'{symbol}    {startDate} to {endDate}')
    # stock = yf.download('AAPL', start='2020-05-22', end='2024-05-22')
    stock = yf.download(symbol, start=startDate, end=endDate)
    close_prices = stock['Close']
    return close_prices


def get_datasets(symbol: str, startDate: str = '2022-04-01', endDate: str = None):
    if not endDate:
        endDate = datetime.now()

    # Ensure yfinance overrides
    yf.pdr_override()

    # Get the stock quote
    df = pdr.get_data_yahoo(symbol, start=startDate, end=endDate)

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close']).values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split the data into training and validation sets
    training_data_len = int(len(scaled_data) * 0.95)
    train_data = scaled_data[:training_data_len]
    valid_data = scaled_data[training_data_len:]

    # Function to create dataset for XGBoost
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    # Define time step and create datasets
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_valid, y_valid = create_dataset(valid_data, time_step)

    return X_train, y_train, X_valid, y_valid, scaler, scaled_data, time_step


def optimize_model(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }

    grid_search = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)

    return grid_search.best_params_


def make_forecast(X_train, y_train, scaler, scaled_data, time_step):
    # Initialize and train the XGBoost model
    xgb_model = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
    xgb_model.fit(X_train, y_train)

    # Predict future prices
    future_prices = []

    # Get the last 60 days of data
    last_60_days = scaled_data[-time_step:].reshape(1, -1)

    for i in range(10):
        pred_price = xgb_model.predict(last_60_days)
        future_prices.append(pred_price[0])
        last_60_days = np.append(last_60_days, pred_price.reshape(1, -1), axis=1)[:, 1:]

    # Inverse transform the predicted prices
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

    return future_prices


def plot_predictions(df):
    plt.plot(df.index[-100:], df['Close'].tail(100), label='Historical Prices')
    plt.plot(pd.date_range(df.index[-1], periods=10), future_prices, linestyle='dashed', color='red', label='Future Predictions')
    plt.title('Stock Price Prediction for Next 10 Days using XGBoost')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.show()


def run_xgboost_forecast(symbol, startDate, endDate = None):
    # XGBoost stock price forecast
    X_train, y_train, X_valid, y_valid, scaler, scaled_data, time_step = get_datasets(symbol, startDate, endDate)

    best_params = optimize_model(X_train, y_train)

    future_prices = make_forecast(X_train, y_train, scaler, scaled_data, time_step)

    # plot_predictions(df)


if __name__ == '__main__':
    run_xgboost_forecast('AMD', '2022-04-01', None)

