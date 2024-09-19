import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import ta
import ta.trend
import ta.momentum
import ta.volume
import ta.volatility
import ta.others
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Principal Component Analysis and Dynamic Time Warping
# https://medium.com/@crisvelasquez/mining-patterns-in-stocks-with-pca-and-dtw-e98651657f37


def add_ta_features(data):
    # Add Trend indicators
    data['trend_ichimoku_conv'] = ta.trend.ichimoku_a(data['High'], data['Low'])
    data['trend_ema_slow'] = ta.trend.ema_indicator(data['Close'], 50)
    data['momentum_kama'] = ta.momentum.kama(data['Close'])
    data['trend_psar_up'] = ta.trend.psar_up(data['High'], data['Low'], data['Close'])
    data['volume_vwap'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'],
                                                               data['Volume']).volume_weighted_average_price()
    data['trend_ichimoku_a'] = ta.trend.ichimoku_a(data['High'], data['Low'])
    data['volatility_kcl'] = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close']).keltner_channel_lband()
    data['trend_ichimoku_b'] = ta.trend.ichimoku_b(data['High'], data['Low'])
    data['trend_ichimoku_base'] = ta.trend.ichimoku_base_line(data['High'], data['Low'])
    data['trend_sma_fast'] = ta.trend.sma_indicator(data['Close'], 20)
    data['volatility_dcm'] = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close']).donchian_channel_mband()
    data['volatility_bbl'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    data['volatility_bbm'] = ta.volatility.BollingerBands(data['Close']).bollinger_mavg()
    data['volatility_kcc'] = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close']).keltner_channel_mband()
    data['volatility_kch'] = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close']).keltner_channel_hband()
    data['trend_sma_slow'] = ta.trend.sma_indicator(data['Close'], 200)
    data['trend_ema_fast'] = ta.trend.ema_indicator(data['Close'], 20)
    data['volatility_dch'] = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close']).donchian_channel_hband()
    data['others_cr'] = ta.others.cumulative_return(data['Close'])
    data['Adj Close'] = data['Close']
    return data


def normalize(ts):
    return (ts - ts.min()) / (ts.max() - ts.min())


def dtw_distance(ts1, ts2, ts1_ta, ts2_ta, weight=0.75):  # Adjust the weight parameter as needed
    ts1_normalized = normalize(ts1)
    ts2_normalized = normalize(ts2)
    distance_pct_change, _ = fastdtw(ts1_normalized.reshape(-1, 1), ts2_normalized.reshape(-1, 1), dist=euclidean)
    distance_ta, _ = fastdtw(ts1_ta, ts2_ta, dist=euclidean)
    distance = weight * distance_pct_change + (1 - weight) * distance_ta
    return distance


def extract_and_reduce_features(data, n_components=3):
    ta_features = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_ta_features = imputer.fit_transform(ta_features)
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(imputed_ta_features)
    return reduced_features


def find_patterns(ticker, data):
    data_with_ta = add_ta_features(data)
    reduced_features = extract_and_reduce_features(data_with_ta, n_components=3)
    price_data_pct_change = data_with_ta['Close'].pct_change().dropna()

    subsequent_days = 15
    days_to = [15, 20, 30]

    min_gap = 10  # Set a minimum gap of 10 days between patterns found

    for n_days in days_to:
        current_window = price_data_pct_change[-n_days:].values
        current_ta_window = reduced_features[-n_days:]

        # Initialize distances with inf
        distances = [np.inf] * (len(price_data_pct_change) - 2 * n_days - subsequent_days)

        for start_index in range(len(price_data_pct_change) - 2 * n_days - subsequent_days):
            # Ensure there is a minimum gap before or after the current window
            gap_before = len(price_data_pct_change) - (start_index + n_days + subsequent_days)
            gap_after = start_index - (len(price_data_pct_change) - n_days)

            if gap_before >= min_gap or gap_after >= min_gap:
                distances[start_index] = dtw_distance(
                    current_window,
                    price_data_pct_change[start_index:start_index + n_days].values,
                    current_ta_window,
                    reduced_features[start_index:start_index + n_days]
                )

        min_distance_indices = np.argsort(distances)[:3]  # find indices of 3 smallest distances
        fig, axs = plt.subplots(1, 2, figsize=(30, 8))

        # plot the entire stock price data
        axs[0].plot(data['Close'], color='blue', label='Overall stock price')

        for i, start_index in enumerate(min_distance_indices):
            # plot the pattern period in different colors
            past_window_start_date = data.index[start_index]
            past_window_end_date = data.index[start_index + n_days + subsequent_days]
            axs[0].plot(data['Close'][past_window_start_date:past_window_end_date], color='C{}'.format(i), label=f"Pattern {i + 1}")

        axs[0].set_title(f'{ticker} Stock Price Data')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Stock Price')
        axs[0].legend()

        # plot the reindexed patterns
        for i, start_index in enumerate(min_distance_indices):
            past_window = price_data_pct_change[start_index:start_index + n_days + subsequent_days]
            reindexed_past_window = (past_window + 1).cumprod() * 100
            axs[1].plot(range(n_days + subsequent_days), reindexed_past_window, color='C{}'.format(i), linewidth=3 if i == 0 else 1,
                        label=f"Past window {i + 1} (with subsequent {subsequent_days} days)")

        reindexed_current_window = (price_data_pct_change[-n_days:] + 1).cumprod() * 100

        axs[1].plot(range(n_days), reindexed_current_window, color='k', linewidth=3, label="Current window")

        # Collect the subsequent prices of the similar patterns
        subsequent_prices = []
        for i, start_index in enumerate(min_distance_indices):
            subsequent_window = price_data_pct_change[start_index + n_days: start_index + n_days + subsequent_days].values
            subsequent_prices.append(subsequent_window)

        subsequent_prices = np.array(subsequent_prices)
        median_subsequent_prices = np.median(subsequent_prices, axis=0)
        # Adjusted line for reindexing the median subsequent prices
        median_subsequent_prices_cum = (median_subsequent_prices + 1).cumprod() * reindexed_current_window.iloc[-1]

        axs[1].plot(range(n_days, n_days + subsequent_days), median_subsequent_prices_cum, color='grey', linestyle='dashed',
                    label="Median Subsequent Price Estimation")

        axs[1].set_title(f"Most similar {n_days}-day patterns in {ticker} stock price history (aligned, reindexed)")
        axs[1].set_xlabel("Days")
        axs[1].set_ylabel("Reindexed Price")
        axs[1].legend()

        # plt.savefig(f'{ticker}_{n_days}_days.png')
        # plt.close(fig)
        plt.show()


def run_pca_dtw(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)
    find_patterns(ticker, data)


if __name__ == "__main__":
    run_pca_dtw('AAPL', '2023-01-01', '2024-06-17')


"""
1.3 Pattern Recognition Method
The aim is to identify patterns in a designated “current window” of price data by comparing it with numerous “past windows” from the historical data. These windows encapsulate a specified number of days, and the process is iterated for different window sizes to identify patterns across various time frames.
For each window, a dual comparison is performed using DTW. One comparison is on the price data while the other is on the PCA-reduced technical indicators data. We compute composite distance measure for each past window vis-a-vis the current window, factoring in both the price data and the technical indicators data.
1.4 Weighting Scheme: PCA vs DTW
We introduce a weight parameter to serve as a fulcrum, balancing the emphasis between the price data and the technical indicators data. By tweaking this parameter, the methodology can be tuned to lean more toward either of the two data sources.

The formula shows the weighted distance computation, where D is the combined DTW distance, Dreturns​ and Dindicators are the DTW distances from percentage change and PCA Reduced indicator features respectively, and ω is the weighting factor.
1.5 Mining Patterns in the Historical Data
The objective is to identify past windows that exhibit a low composite distance to the current window, indicating a high degree of similarity. The lower the distance, the more analogous the patterns.
The three to five most similar past windows are then retrieved and visualized, and their subsequent price movements are analyzed to project a median subsequent price path for the current window.

4. Limitations and Improvements
This has only been the first step in utilizing a combination of DTW and PCA for pattern recognition in stock price data. As a consequence, there are certain limitations and areas for improvement that should be considered.
4.1 Limitations
4.1.1. Optimization of Parameter through Backtesting
The performance of the DTW and PCA techniques significantly depends on the selection of parameters such as the weighting factor in the distance computation, the window sizes for days to window, and the variables chosen in PCA. Currently, there isn’t a systematic approach for parameter tuning.
Incorporating automated parameter tuning techniques like grid search or evolutionary algorithms could help in finding the optimal set of parameters that maximize the accuracy and reliability of the pattern recognition and projection methodology
4.1.2. Assumption of Repeating Patterns
The core assumption that past price patterns will reoccur in the future is a common simplification, yet it’s not always valid. Market dynamics evolve, and historical patterns may not necessarily repeat or lead to accurate projections.
4.2 Further Improvements
4.2.1. Incorporation of Additional Predictive Indicators
Extending the feature set by incorporating more diverse predictive indicators or alternative data sources could potentially enhance the accuracy of the pattern recognition methodology.
4.2.2. Statistical Validation of Patterns
An improvement could be making sure the patterns found are truly similar, not just by looking, but using a statistical test like a t-test. This test helps us see if the patterns we find in historical data are statistically similar to the current pattern or if they just look alike by chance.
4.2.3. Robustness Against Market Anomalies
Designing mechanisms to identify and adjust for price anomalies or extreme events could improve the resilience and accuracy of the pattern recognition and projection methodology. One way to smoothen out anomalies is through moving average methods.
36 Moving Average Methods in Python For Stock Price Analysis [1/4]
The Fundamentals — SMA, EMA, WMA, KAMA and Their Nuances
medium.com
4.2.4. Cross-Asset Pattern Recognition
Expanding the scope to include similar assets or even different markets could also provide a richer analysis. By looking at patterns across related assets or sectors, we might uncover broader market trends or recurring patterns that are not just limited to the individual asset initially analyzed.
"""

