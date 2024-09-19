import ccxt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
import tf2onnx
import matplotlib.pyplot as plt
import numpy as np


# Binance data download feature
def download_data(ticker, timeframe='1d', start_date='2004-01-01T00:00:00Z', end_date='2024-01-01T00:00:00Z'):
    exchange = ccxt.binance({'enableRateLimit': False})
    since = exchange.parse8601(start_date)
    end_date_timestamp = pd.to_datetime(end_date, utc=True)
    all_data = []

    while since < end_date_timestamp.timestamp() * 1000:
        ohlc = exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since)
        all_data.extend(ohlc)
        since = ohlc[-1][0] + 1  # increment the `since` parameter by one millisecond

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Verify both are timezone-aware or convert if necessary
    if df.index.tz is None:
        df.index = df.index.tz_localize('utc')

    df = df[df.index <= end_date_timestamp]
    print(df)
    return df['close'].values


def download_normalized_data(ticker: str):
    # Load data
    # data = download_data('ETH/USDT')
    data = download_data(ticker)
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    return data


# Function to create samples from the sequence
def create_samples(dataset, time_steps=120):
    X, y = [], []
    for i in range(time_steps, len(dataset)):
        X.append(dataset[i - time_steps:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


def prepare_training_and_test(data, time_steps=120):
    # Prepare training and test data
    # time_steps = 120
    X, y = create_samples(data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Split the data (80% for training)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the model
    # model = tf.keras.models.Sequential()
    model = tf.keras.Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Set up early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    )
    # Checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )

    # model training for 300 epochs
    history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test),
                        batch_size=32, callbacks=[early_stopping, checkpoint], verbose=2)
    return model, X_train, y_train, X_test, y_test, history


# Remove any symbols that are not valid for filenames
def to_filename(ticker: str, lower_case=False):
    t = ticker.replace('/', '')
    t = t.replace('-', '')
    if lower_case:
        return t.lower()
    else:
        return t


def plot_training_history(ticker: str, history):
    # Plot the training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['rmse'], label='Train RMSE')
    plt.plot(history.history['val_rmse'], label='Validation RMSE')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/RMSE')
    plt.legend()
    plt.savefig(f'{to_filename(ticker)}.png')  # Save the plot as an image file


def convert_model_to_onnx(ticker, model):
    # Convert the model to ONNX
    symbol = to_filename(ticker, True)
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=f"model_{symbol}.onnx")
    print(f"ONNX model saved as 'model_{symbol}.onnx'.")
    return onnx_model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Evaluate the model
    train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"train_loss={train_loss:.3f}, train_rmse={train_rmse:.3f}")
    print(f"test_loss={test_loss:.3f}, test_rmse={test_rmse:.3f}")


def run_deep_learning_strategy_model(ticker: str):
    # ticker = 'ETH/USDT'
    data = download_normalized_data(ticker)
    model, X_train, y_train, X_test, y_test, history = prepare_training_and_test(data)
    plot_training_history(ticker, history)

    onnx_model = convert_model_to_onnx(ticker, model)
    evaluate_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":

    run_deep_learning_strategy_model('ETH/USDT')
