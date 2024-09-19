import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import savgol_filter
import plotly.express as px
from statsforecast import StatsForecast

# https://medium.com/bip-xtech/stop-using-moving-average-to-smooth-your-time-series-2179af9ed59b


def read_data():
    train = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv')
    test = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv').rename(columns={'y': 'y_test'})
    uid = np.array(['H386'])
    df_train = train.query('unique_id in @uid')
    df_test = test.query('unique_id in @uid')
    StatsForecast.plot(df_train, df_test, plot_random = False, engine='plotly')

    computed_features = [] # I will need this list to plot later the smoothed series
    for window_size in [10, 25]:
        df_train.loc[:,f'moving_average_{window_size}'] = df_train['y'].rolling(window=window_size, center=True).mean()
        df_train.loc[:,f'savgol_filter_{window_size}'] = savgol_filter(df_train['y'], window_size, 2)
        computed_features.append(f'moving_average_{window_size}')
        computed_features.append(f'savgol_filter_{window_size}')

    return df_train, computed_features


def plot_savitzky_golay(df_train, computed_features):
    fig = px.line(df_train[df_train.ds>500], x='ds', y=['y'] + computed_features[:2], title='Different moving average estimators',
                  labels={'Value': 'y', 'Date': 'Date'},
                  line_shape='linear')

    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sensor Value',
        hovermode='x'
    )

    fig.show()


# Windows size 25
def plot_savitzky_golay_25(df_train, computed_features):
    fig = px.line(df_train[df_train.ds > 500], x='ds', y=['y'] + computed_features[2:4], title='Different moving average estimators',
                  labels={'Value': 'y', 'Date': 'Date'},
                  line_shape='linear')

    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sensor Value',
        hovermode='x'
    )

    fig.show()


def run_savitzky_golay():
    df_train, computed_features = read_data()
    plot_savitzky_golay(df_train, computed_features)
    plot_savitzky_golay_25(df_train, computed_features)


if __name__ == "__main__":

    run_savitzky_golay()

