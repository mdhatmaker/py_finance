import numpy as np
import matplotlib.pyplot as plt
from api.utils.yahoo_finance import download_close_prices

# https://www.askpython.com/python/examples/adx-indicator-python


def calculate_directional_movement(prices):
    dm_pos = []
    dm_neg = []
    for i in range(1, len(prices)):
        price_diff = prices[i] - prices[i - 1]
        if price_diff > 0:
            dm_pos.append(price_diff)
            dm_neg.append(0)
        elif price_diff < 0:
            dm_pos.append(0)
            dm_neg.append(-price_diff)
        else:
            dm_pos.append(0)
            dm_neg.append(0)
    return dm_pos, dm_neg


def calculate_true_range(prices):
    true_ranges = []
    for i in range(1, len(prices)):
        high_low = prices[i] - prices[i - 1]
        high_close = abs(prices[i] - prices[i - 1])
        low_close = abs(prices[i] - prices[i - 1])
        true_ranges.append(max(high_low, high_close, low_close))
    return true_ranges


def calculate_directional_index(dm_pos, dm_neg, true_ranges, window):
    atr = [np.mean(true_ranges[:window])]
    di_pos = [np.mean(dm_pos[:window])]
    di_neg = [np.mean(dm_neg[:window])]
    for i in range(window, len(dm_pos)):
        atr.append((atr[-1] * (window - 1) + true_ranges[i]) / window)
        di_pos.append((di_pos[-1] * (window - 1) + dm_pos[i]) / window)
        di_neg.append((di_neg[-1] * (window - 1) + dm_neg[i]) / window)
    di_pos = np.array(di_pos)
    di_neg = np.array(di_neg)
    dx = np.abs((di_pos - di_neg) / (di_pos + di_neg)) * 100
    adx = [np.mean(dx[:window])]
    for i in range(window, len(dx)):
        adx.append((adx[-1] * (window - 1) + dx[i]) / window)
    return adx


# def generate_stock_prices(num_prices=100, initial_price=100, volatility=0.05):
#     prices = [initial_price]
#     for _ in range(1, num_prices):
#         price_change = np.random.normal(0, volatility)
#         prices.append(prices[-1] + price_change * prices[-1])
#     return prices


def run_adx_indicator(ticker, start_date, end_date, window=14):
    # stock_prices = generate_stock_prices()
    stock_prices = download_close_prices(ticker, start_date, end_date)
    stock_prices = np.asarray(stock_prices)

    # Calculate ADX
    # window = 14  # ADX window
    dm_pos, dm_neg = calculate_directional_movement(stock_prices)
    true_ranges = calculate_true_range(stock_prices)
    adx = calculate_directional_index(dm_pos, dm_neg, true_ranges, window)

    plot_adx_indicator(ticker, stock_prices, adx)


def plot_adx_indicator(ticker, stock_prices, adx):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(stock_prices, label='Stock Prices')
    plt.title(f'{ticker} Stock Prices')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(adx, label='ADX', color='red')
    plt.axhline(y=20, color='gray', linestyle='--')
    plt.axhline(y=50, color='gray', linestyle='--')
    plt.axhline(y=70, color='gray', linestyle='--')
    plt.title('ADX')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    run_adx_indicator('AAPL', '2023-01-01', None)

