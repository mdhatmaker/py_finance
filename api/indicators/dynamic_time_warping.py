import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from pandas import Series
from scipy.spatial.distance import euclidean
from api.utils.yahoo_finance import download_close_prices, download_ohlc, download_returns


# https://medium.com/@crisvelasquez/pattern-mining-for-stock-prediction-with-dynamic-time-warping-3f8df5fb4c5b


def normalize(ts):
    return (ts - ts.min()) / (ts.max() - ts.min())


def dtw_distance(ts1, ts2):
    ts1_normalized = normalize(ts1)
    ts2_normalized = normalize(ts2)
    distance, _ = fastdtw(ts1_normalized.reshape(-1, 1), ts2_normalized.reshape(-1, 1), dist=euclidean)
    return distance


def find_most_similar_pattern(n_days: int, subsequent_days: int, price_data_pct_change: Series):
    current_window = price_data_pct_change[-n_days:].values
    # Adjust to find and store 5 patterns
    min_distances = [(float('inf'), -1) for _ in range(5)]
    for start_index in range(len(price_data_pct_change) - 2 * n_days - subsequent_days):
        past_window = price_data_pct_change[start_index:start_index + n_days].values
        distance = dtw_distance(current_window, past_window)
        for i, (min_distance, _) in enumerate(min_distances):
            if distance < min_distance:
                min_distances[i] = (distance, start_index)
                break
    return min_distances


# data = yf.download(ticker, start=start_date, end=end_date)
# Transform price data into returns
# price_data = data['Close']
# price_data_pct_change = price_data.pct_change().dropna()


def generate_time_warping_plots(ticker, start_date, end_date):
    price_data = download_close_prices(ticker, start_date, end_date)
    price_data_pct_change = download_returns(ticker, start_date, end_date)

    # Different Windows to find patterns on,
    # e.g. if 15, The code will find the most similar 15 day in the history
    days_to = [15, 20, 30]

    # Number of days for price development observation
    # e.g. if 20, then the subsequent 20 days after pattern window is found will be plotted
    subsequent_days = 20

    for n_days in days_to:
        min_distances = find_most_similar_pattern(n_days, subsequent_days, price_data_pct_change)
        fig, axs = plt.subplots(1, 2, figsize=(30, 6))
        axs[0].plot(price_data, color='blue', label='Overall stock price')
        color_cycle = ['red', 'green', 'purple', 'orange', 'cyan']
        subsequent_prices = []

        for i, (_, start_index) in enumerate(min_distances):
            color = color_cycle[i % len(color_cycle)]
            past_window_start_date = price_data.index[start_index]
            past_window_end_date = price_data.index[start_index + n_days + subsequent_days]
            axs[0].plot(price_data[past_window_start_date:past_window_end_date], color=color, label=f"Pattern {i + 1}")
            # Store subsequent prices for median calculation
            subsequent_window = price_data_pct_change[start_index + n_days: start_index + n_days + subsequent_days].values
            subsequent_prices.append(subsequent_window)

        axs[0].set_title(f'{ticker} Stock Price Data')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Stock Price')
        axs[0].legend()

        for i, (_, start_index) in enumerate(min_distances):
            color = color_cycle[i % len(color_cycle)]
            past_window = price_data_pct_change[start_index:start_index + n_days + subsequent_days]
            reindexed_past_window = (past_window + 1).cumprod() * 100
            axs[1].plot(range(n_days + subsequent_days), reindexed_past_window, color=color, linewidth=3 if i == 0 else 1,
                        label=f"Past window {i + 1} (with subsequent {subsequent_days} days)")

        reindexed_current_window = (price_data_pct_change[-n_days:] + 1).cumprod() * 100
        axs[1].plot(range(n_days), reindexed_current_window, color='k', linewidth=3, label="Current window")

        # Compute and plot the median subsequent prices
        subsequent_prices = np.array(subsequent_prices)
        median_subsequent_prices = np.median(subsequent_prices, axis=0)
        median_subsequent_prices_cum = (median_subsequent_prices + 1).cumprod() * reindexed_current_window.iloc[-1]

        axs[1].plot(range(n_days, n_days + subsequent_days), median_subsequent_prices_cum, color='black', linestyle='dashed',
                    label="Median Subsequent Price Estimation")
        axs[1].set_title(f"Most similar {n_days}-day patterns in {ticker} stock price history (aligned, reindexed)")
        axs[1].set_xlabel("Days")
        axs[1].set_ylabel("Reindexed Price")
        axs[1].legend()

        plt.show()


if __name__ == "__main__":

    ticker = "ASML.AS"
    start_date = '2000-01-01'
    end_date = '2023-07-21'

    generate_time_warping_plots(ticker, start_date, end_date)


"""
1. What is Dynamic Time Warping?
Originating from speech recognition efforts, Dynamic Time Warping brings a specialized lens to time series analysis, adeptly identifying similarities between temporal sequences of varying speeds and timings. Whilst initially employed to comprehend variations in spoken words, DTW has found utility in financial domains.
Consider two temporal sequences, A and B, where A = a1​, a2​,…, an​ and B = b1​, b2​,…, bm​. Traditional metrics like Euclidean distance may inaccurately represent the true similarity between A and B if they are out of phase. Formally, the Euclidean distance is given by:

Equation 1. The Euclidean distance DE​ between two time series A and B of equal length n, where ai​ and bi​ are the ith elements of A and B respectively. The Euclidean distance computes the straight-line distance between corresponding points of the time series.
Whereas, DTW offers flexibility in aligning the sequences in a non-linear fashion, minimizing the cumulative distance and therefore, rendering a more accurate representation of their similarity. The DTW distance between A and B is computed as:

Equation 2. The DTW distance between two time series A and B of potentially unequal length. ai​ and bj(i)​ denote the elements of A and B at arbitrary indices i and j(i) respectively. DTW allows for optimal alignment between the points of the two time series, thereby enabling a more flexible and context-aware measure of similarity.
Where j (i) represents an alignment function that finds the optimal alignment between elements of A and B, minimizing the cumulative distance. DTW effectively warps the time dimension to align the sequences, ensuring each point in sequence A is matched with the most analogous point in sequence B.
The pattern of stock prices, propelled by a myriad of factors, generates time series that frequently embodies patterns that could hint at future movements. While some patterns appear overtly similar, others may be subtly analogous with differences in their timing or amplitude.
DTW allows us to ascertain the similarity between a current price pattern and historical patterns by aligning them optimally in the time dimension. This facilitates understanding of how current market dynamics mirror historical instances, providing insights that can potentially forecast upcoming price movements.

4. Limitations and Improvements
The ability to identify recurring patterns in stock price movements is vital for insightful analysis and informed investment decision-making. While the application of DTW in the context of stock price pattern recognition, as demonstrated in this article, is compelling, it also reveals certain limitations and opens avenues for improvements and refinements

4.1 Limitations
Univariate Analysis: The current implementation of DTW focuses predominantly on stock prices, overlooking other influential variables like trading volumes or volatility, which can provide vital insights into market behavior and augment predictive capabilities. Incorporating these additional aspects could furnish a more holistic approach to understanding stock price movements.
Computational Intensity: Especially with larger datasets, DTW can be computationally intense and inefficient due to its quadratic time complexity. For vast historical stock price data, this might manifest as prolonged processing times and significant resource usage.
Assumption of Similar Patterns: DTW presupposes that historical patterns will manifest in the future, an assumption that will not always hold true in the volatile and multifaceted stock market, influenced by numerous, occasionally unprecedented, variables.
Sensitivity to Noise: The sensitivity of DTW to noise within the data can potentially impact the precision of pattern identification, where transient fluctuations might unduly influence the DTW distance calculation. One way to tackle noise and anomalous spikes is by smoothing out the data through the utilization of moving averages
36 Moving Average Methods in Python For Stock Price Analysis [1/4]
The Fundamentals — SMA, EMA, WMA, KAMA and Their Nuances
medium.com

4.2 Improvements
Optimization Techniques: Employing optimization methods, like FastDTW, which reduces time complexity by approximating DTW distances, can be pivotal in managing computational demands.
Noise Reduction: Implementing noise reduction methodologies can isolate substantial patterns by diminishing the impact of transient data irregularities.
Feature Engineering: The inclusion of auxiliary features might amplify the robustness and insightfulness of pattern recognition.
Machine Learning Integration: Coupling DTW with machine learning models can enhance predictive and pattern recognition capabilities by amalgamating detailed pattern identification with predictive validation.
Multivariate DTW: Considering multivariate DTW can provide a multi-faceted analytical framework by assimilating multiple variables simultaneously.
"""
