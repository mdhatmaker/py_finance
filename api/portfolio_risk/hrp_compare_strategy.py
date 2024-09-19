import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import riskfolio as rp
import warnings

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format


# Tickers of top companies from S&P 500
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'WMT', 'V', 'JPM',
          'PG', 'UNH', 'DIS', 'NVDA', 'HD', 'PYPL', 'MA', 'BAC', 'INTC', 'T']
assets.sort()


def download_data(start, end):
    # Downloading data
    data = yf.download(assets, start=start, end=end)['Adj Close']
    data.columns = assets
    return data


def calculate_returns(data, start, end):
    # Calculating returns for HRP
    Y = data[assets].pct_change().dropna()

    # HRP Portfolio Optimization
    port = rp.HCPortfolio(returns=Y)
    w = port.optimization(model='HRP', codependence='pearson', rm='MV', rf=0, linkage='single', max_k=10, leaf_order=True)

    # Calculate Daily Returns for the HRP Portfolio
    daily_portfolio_returns = data.pct_change().dot(w.mean(axis=1))

    # Download S&P 500 Data for Benchmark
    sp500 = yf.download('^GSPC', start=start, end=end)['Adj Close']
    benchmark_returns = sp500.pct_change().dropna()

    # Align dates
    daily_portfolio_returns, benchmark_returns = daily_portfolio_returns.align(benchmark_returns, join='inner', axis=0)

    # Calculate Equal Weights for each asset
    equal_weights = np.repeat(1/len(assets), len(assets))

    # Calculate Daily Returns for the Equal Weighted Portfolio
    daily_equal_returns = data.pct_change().dot(equal_weights)

    # Align dates for Equal Weighted Portfolio
    daily_equal_returns = daily_equal_returns.reindex(daily_portfolio_returns.index)

    # Performance Metrics for Equal Weighted Portfolio
    cumulative_equal_returns = (1 + daily_equal_returns).cumprod()
    annualized_equal_return = np.mean(daily_equal_returns) * 252
    volatility_equal = np.std(daily_equal_returns) * np.sqrt(252)
    sharpe_ratio_equal = annualized_equal_return / volatility_equal
    # Printing Metrics for Equal Weighted Portfolio
    print("\nEqual Weighted Portfolio Metrics:")
    print(f"Annualized Return: {annualized_equal_return:.2%}")
    print(f"Volatility: {volatility_equal:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_equal:.2f}")

    # Calculate Performance Metrics for HRP Portfolio and Benchmark
    cumulative_portfolio_returns = (1 + daily_portfolio_returns).cumprod()
    cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()
    annualized_portfolio_return = np.mean(daily_portfolio_returns) * 252
    volatility_portfolio = np.std(daily_portfolio_returns) * np.sqrt(252)
    sharpe_ratio_portfolio = annualized_portfolio_return / volatility_portfolio
    annualized_benchmark_return = np.mean(benchmark_returns) * 252
    volatility_benchmark = np.std(benchmark_returns) * np.sqrt(252)
    sharpe_ratio_benchmark = annualized_benchmark_return / volatility_benchmark
    # Print the metrics for HRP Portfolio and S&P 500 Benchmark
    print("\nHRP Portfolio Metrics:")
    print(f"Annualized Return: {annualized_portfolio_return:.2%}")
    print(f"Volatility: {volatility_portfolio:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_portfolio:.2f}")

    print("\nS&P 500 Benchmark Metrics:")
    print(f"Annualized Return: {annualized_benchmark_return:.2%}")
    print(f"Volatility: {volatility_benchmark:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_benchmark:.2f}")

    return cumulative_equal_returns, cumulative_portfolio_returns, cumulative_benchmark_returns


def plot_hrp_compare_strategy(cumulative_equal_returns, cumulative_portfolio_returns, cumulative_benchmark_returns):
    # Plotting Comparative Performance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_portfolio_returns, label='HRP Portfolio')
    plt.plot(cumulative_equal_returns, label='Equal Weighted Portfolio')
    plt.plot(cumulative_benchmark_returns, label='S&P 500 Benchmark')
    plt.title('Comparative Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()


def run_hrp_compare_strategy(start, end):
    # start = '2018-01-01'
    # end = '2024-01-01'
    data = download_data(start, end)
    cum_equal, cum_portfolio, cum_benchmark = calculate_returns(data, start, end)
    plot_hrp_compare_strategy(cum_equal, cum_portfolio, cum_benchmark)


if __name__ == "__main__":

    run_hrp_compare_strategy('2018-01-01', '2024-01-01')
