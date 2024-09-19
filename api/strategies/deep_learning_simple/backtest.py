from typing import Any
import ccxt
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


class ErrorMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float

    def __init__(self, mae: float, rmse: float, r2: float, mape: float):
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2
        self.mape = mape


class StrategyMetrics:
    sharp_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float

    def __init__(self, sharp_ratio: float, sortino_ratio: float, beta: float, alpha: float):
        self.sharp_ratio = sharp_ratio
        self.sortino_ratio = sortino_ratio
        self.beta = beta
        self.alpha = alpha


def download_data(symbol: str, interval: str, limit: int = 1000) -> tuple[DataFrame, Any, Any]:
    # Create an instance of the Binance exchange
    binance = ccxt.binance()

    # Define the market symbol and time interval
    # symbol = 'ETH/USDT'
    # interval = '1d'
    # limit = 1000  # Download enough data for 120-day windows

    # Download the historical data
    ohlcv = binance.fetch_ohlcv(symbol, interval, limit=limit)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save data to CSV file
    df.to_csv('binance_data.csv', index=False)
    print("Data downloaded and saved in 'binance_data.csv'")

    # Download the data
    data = pd.read_csv('binance_data.csv')

    # Make sure the timestamp column is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Normalization values (adjust these values to the ones used to normalize)
    min_close = data['close'].min()
    max_close = data['close'].max()

    # Normalize closing data
    data['close_normalized'] = (data['close'] - min_close) / (max_close - min_close)

    return data, min_close, max_close


# Remove any symbols that are not valid for filenames
def to_filename(ticker: str, lower_case=False):
    t = ticker.replace('/', '_')
    t = t.replace('-', '_')
    if lower_case:
        return t.lower()
    else:
        return t


def generate_predictions(ticker: str, data: DataFrame, start_date_str: str, min_close, max_close, sequence_length: int = 120):  # Adjust sequence_length based on the model
    symbol = to_filename(ticker, True)
    # Load the ONNX model
    model = onnx.load(f'model_{symbol}.onnx')
    onnx.checker.check_model(model)

    # Create a runtime session
    ort_session = ort.InferenceSession(f'model_{symbol}.onnx')

    # Prepare data for the model as sliding windows
    input_name = ort_session.get_inputs()[0].name
    # sequence_length = 120

    # Create a list to store predications
    predictions_list = []

    # Define the start date for predictions
    # start_date = pd.Timestamp('2024-01-01')
    start_date = pd.Timestamp(start_date_str)
    end_date = pd.Timestamp.today()

    # Perform inference day by day
    current_date = start_date
    while current_date <= end_date:
        # Select the last 120 days of data before the current date
        end_idx = data[data['timestamp'] <= current_date].index[-1]
        start_idx = end_idx - sequence_length + 1

        if start_idx < 0:
            print(f"There is not sufficient data for the date {current_date}")
            break

        # Extract the Normalized Data Window
        window = data['close_normalized'].values[start_idx:end_idx + 1]

        if len(window) < sequence_length:
            print(f"There is not sufficient data for the date {current_date}")
            break

        # Prepare the data for the model
        input_window = np.array(window).astype(np.float32)
        input_window = np.expand_dims(input_window, axis=0)  # Add batch size dimension
        input_window = np.expand_dims(input_window, axis=2)  # Add feature dimension

        # Performing the inference
        output = ort_session.run(None, {input_name: input_window})
        prediction = output[0][0][0]

        # Denormalize the prediction
        prediction = prediction * (max_close - min_close) + min_close

        # Store the prediction
        predictions_list.append({'date': current_date, 'prediction': prediction})

        # Increment the date
        current_date += pd.Timedelta(days=1)

    # Convert the list of predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions_list)

    # Save predictions to a CSV file
    predictions_df.to_csv('predicted_data.csv', index=False)
    print("Predictions saved in 'predicted_data.csv'")

    # Compare predictions with real values
    comparison_df = pd.merge(predictions_df, data[['timestamp', 'close']], left_on='date', right_on='timestamp')
    comparison_df = comparison_df.drop(columns=['timestamp'])
    comparison_df = comparison_df.rename(columns={'close': 'actual'})

    # Calculate error metrics
    mae = mean_absolute_error(comparison_df['actual'], comparison_df['prediction'])
    rmse = np.sqrt(mean_squared_error(comparison_df['actual'], comparison_df['prediction']))
    r2 = r2_score(comparison_df['actual'], comparison_df['prediction'])
    mape = mean_absolute_percentage_error(comparison_df['actual'], comparison_df['prediction'])
    metrics = ErrorMetrics(mae, rmse, r2, mape)
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    return ort_session, predictions_df, comparison_df, metrics


def plot_with_error_bands(ticker: str, comparison_df, mae):
    # Drawing the Graph with Error Bands
    plt.figure(figsize=(14, 7))
    plt.plot(comparison_df['date'], comparison_df['actual'], label='Actual Price', color='blue')
    plt.plot(comparison_df['date'], comparison_df['prediction'], label='Predicted Price', color='red')
    plt.fill_between(comparison_df['date'], comparison_df['prediction'] - mae, comparison_df['prediction'] + mae,
                     color='gray', alpha=0.2, label='Error Band (MAE)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Price Prediction vs Actual')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_price_prediction.png")
    plt.show()
    print(f"Graph saved as '{to_filename(ticker)}_price_prediction.png'")

    # Residual error analysis
    residuals = comparison_df['actual'] - comparison_df['prediction']
    plt.figure(figsize=(14, 7))
    plt.plot(comparison_df['date'], residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.title(f'{ticker} Prediction Residuals')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_residuals.png")
    plt.show()
    print(f"Residual graph saved as '{to_filename(ticker)}_residuals.png'")


def calculate_correlation(comparison_df):
    # Correlation analysis
    correlation = comparison_df['actual'].corr(comparison_df['prediction'])
    print(f'Correlation between actual and predicted prices: {correlation}')
    return correlation


def calculate_returns(comparison_df):
    # Prediction-based investment strategy simulation (original strategy)
    investment_df = comparison_df.copy()
    investment_df['strategy_returns'] = (investment_df['prediction'].shift(-1) - investment_df['actual']) / investment_df['actual']
    investment_df['buy_and_hold_returns'] = (investment_df['actual'].shift(-1) - investment_df['actual']) / investment_df['actual']
    strategy_cumulative_returns = (investment_df['strategy_returns'] + 1).cumprod() - 1
    buy_and_hold_cumulative_returns = (investment_df['buy_and_hold_returns'] + 1).cumprod() - 1
    return investment_df, strategy_cumulative_returns, buy_and_hold_cumulative_returns


def plot_cumulative_returns(ticker, investment_df, strategy_cumulative_returns, buy_and_hold_cumulative_returns):
    plt.figure(figsize=(14, 7))
    plt.plot(investment_df['date'], strategy_cumulative_returns, label='Strategy Cumulative Returns', color='green')
    plt.plot(investment_df['date'], buy_and_hold_cumulative_returns, label='Buy and Hold Cumulative Returns', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(f'{ticker} Investment Strategy vs Buy and Hold')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_investment_strategy.png")
    plt.show()
    print(f"Investment strategy chart saved as '{to_filename(ticker)}_investment_strategy.png'")


def calculate_drawdown_analysis(investment_df, strategy_cumulative_returns):
    # Drawdown analysis
    investment_df['drawdown'] = strategy_cumulative_returns.cummax() - strategy_cumulative_returns
    investment_df['max_drawdown'] = investment_df['drawdown'].max()
    return investment_df


def plot_drawdown_analysis(ticker, investment_df):
    plt.figure(figsize=(14, 7))
    plt.plot(investment_df['date'], investment_df['drawdown'], label='Drawdown', color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title(f'{ticker} Strategy Drawdown')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_drawdown.png")
    plt.show()
    print(f"Drawdown graph saved as '{to_filename(ticker)}_drawdown.png'")


def calculate_return_metrics(investment_df, risk_free_rate = 0.01):     # Assumes an annual risk-free rate of 1%
    # Sharpe ratio of the original strategy
    strategy_returns_daily = investment_df['strategy_returns'].dropna()
    excess_returns = strategy_returns_daily - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    print(f'Sharpe Ratio (original strategy): {sharpe_ratio}')
    # Implementation of the new strategy
    investment_df['position'] = np.where(investment_df['prediction'].shift(-1) > investment_df['prediction'], 1, -1)
    investment_df['strategy_returns_new'] = investment_df['position'] * investment_df['buy_and_hold_returns']
    strategy_cumulative_returns_new = (investment_df['strategy_returns_new'] + 1).cumprod() - 1
    return investment_df, strategy_cumulative_returns_new


def plot_new_strategy(ticker: str, investment_df, strategy_cumulative_returns_new, buy_and_hold_cumulative_returns):
    # Drawing the new strategy chart
    plt.figure(figsize=(14, 7))
    plt.plot(investment_df['date'], strategy_cumulative_returns_new, label='New Strategy Cumulative Returns', color='blue')
    plt.plot(investment_df['date'], buy_and_hold_cumulative_returns, label='Buy and Hold Cumulative Returns', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(f'{ticker} New Investment Strategy vs Buy and Hold')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_new_investment_strategy.png")
    plt.show()
    print(f"New investment strategy chart saved as '{to_filename(ticker)}_new_investment_strategy.png'")


def calculate_new_strategy_metrics(investment_df, risk_free_rate = 0.01):   # Assumes an annual risk-free rate of 1%
    # Sharpe Ratio of the new strategy
    strategy_returns_daily_new = investment_df['strategy_returns_new'].dropna()
    excess_returns_new = strategy_returns_daily_new - risk_free_rate / 252
    sharpe_ratio_new = np.mean(excess_returns_new) / np.std(excess_returns_new) * np.sqrt(252)
    print(f'Sharpe Ratio (New Strategy): {sharpe_ratio_new}')
    # Calculate additional metrics: Sortino, Beta, and Alpha index
    # Sortino Index
    downside_returns = strategy_returns_daily_new[strategy_returns_daily_new < 0]
    sortino_ratio = np.mean(excess_returns_new) / np.std(downside_returns) * np.sqrt(252)
    print(f'Sortino Ratio (New Strategy): {sortino_ratio}')
    # Beta and Alpha
    market_returns = investment_df['buy_and_hold_returns'].dropna()
    covariance_matrix = np.cov(strategy_returns_daily_new, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    alpha = np.mean(strategy_returns_daily_new) - beta * np.mean(market_returns)
    print(f'Beta (New Strategy): {beta}')
    print(f'Alpha (New Strategy): {alpha}')
    return StrategyMetrics(sharpe_ratio_new, sortino_ratio, beta, alpha)


def cross_validation(data, min_close, max_close, ort_session, sequence_length: int = 120):
    input_name = ort_session.get_inputs()[0].name
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_scores = []
    for train_index, test_index in tscv.split(data):
        train = data.loc[train_index]
        test = data.loc[test_index]
        train.loc[:, 'close_normalized'] = (train['close'] - min_close) / (max_close - min_close)
        test.loc[:, 'close_normalized'] = (test['close'] - min_close) / (max_close - min_close)

        predictions_cv = []
        for i in range(len(test) - sequence_length):
            input_window = train['close_normalized'].values[-sequence_length + i:]
            input_window = np.append(input_window, test['close_normalized'].values[:i + 1])
            input_window = np.array(input_window[-sequence_length:]).astype(np.float32)
            input_window = np.expand_dims(input_window, axis=0)  # Add batch size dimension
            input_window = np.expand_dims(input_window, axis=2)  # Add feature dimension

            output = ort_session.run(None, {input_name: input_window})
            prediction = output[0][0][0]
            prediction = prediction * (max_close - min_close) + min_close
            predictions_cv.append(prediction)

        actuals_cv = test['close'].values[sequence_length:]
        mae_cv = mean_absolute_error(actuals_cv, predictions_cv)
        cross_val_scores.append(mae_cv)
    print(f'Cross-Validation MAE: {np.mean(cross_val_scores)} ± {np.std(cross_val_scores)}')

    # Comparison with other models
    # Here you can add the comparison with other models, such as a moving average model

    # Simple Moving Average (SMA) Model
    data['SMA'] = data['close'].rolling(window=sequence_length).mean()
    # Moving Average Model Predictions
    data = data.dropna()
    sma_predictions = data['SMA'].values
    sma_actuals = data['close'].values

    sma_mae = mean_absolute_error(sma_actuals, sma_predictions)
    sma_rmse = np.sqrt(mean_squared_error(sma_actuals, sma_predictions))
    sma_r2 = r2_score(sma_actuals, sma_predictions)
    metrics = ErrorMetrics(sma_mae, sma_rmse, sma_r2, np.nan)
    print(f'SMA Mean Absolute Error (MAE): {sma_mae}')
    print(f'SMA Root Mean Squared Error (RMSE): {sma_rmse}')
    print(f'SMA R-squared (R2): {sma_r2}')
    return data, metrics


def plot_sma(ticker, data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['timestamp'], data['close'], label='Actual Price', color='blue')
    plt.plot(data['timestamp'], data['SMA'], label='SMA Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} SMA Price Prediction vs Actual')
    plt.legend()
    plt.savefig(f"{to_filename(ticker)}_sma_price_prediction.png")
    plt.show()
    print(f"SMA Prediction Graph saved as '{to_filename(ticker)}_sma_price_prediction.png'")


def run_deep_learning_strategy_backtest(ticker: str, start_date: str, interval: str = '1d', risk_free_rate = 0.01, sequence_length = 120):
    # ticker = ''ETH/USDT'
    # start_date_str = '2020-01-01'
    # interval = '1d'
    # risk_free_rate = 0.01
    # sequence_length = 120

    data, min_close, max_close = download_data(ticker, interval)
    ort_session, predictions_df, comparison_df, metrics = generate_predictions(ticker, data, start_date, min_close, max_close, sequence_length)
    plot_with_error_bands(ticker, comparison_df, metrics.mae)

    correlation = calculate_correlation(comparison_df)
    investment_df, strategy_cumret, buy_and_hold_cumret = calculate_returns(comparison_df)
    plot_cumulative_returns(ticker, investment_df, strategy_cumret, buy_and_hold_cumret)

    investment_df = calculate_drawdown_analysis(investment_df, strategy_cumret)
    plot_drawdown_analysis(ticker, investment_df)

    investment_df, strategy_cumret_new = calculate_return_metrics(investment_df, risk_free_rate)
    plot_new_strategy(ticker, investment_df, strategy_cumret_new, buy_and_hold_cumret)

    metrics_new = calculate_new_strategy_metrics(investment_df, risk_free_rate)

    data, sma_metrics = cross_validation(data, min_close, max_close, ort_session, sequence_length)
    plot_sma(ticker, data)


if __name__ == "__main__":

    run_deep_learning_strategy_backtest('ETH/USDT', '1d', '2022-01-01', risk_free_rate=0.01)

