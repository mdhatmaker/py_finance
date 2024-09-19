import warnings
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt
import itertools
import numpy as np

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format


# Tickers of top companies from S&P 500
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'ASML', 'NVDA', 'UNH', 'V']
assets.sort()


def download_data(start, end):
    # Date range
    start = '2018-01-01'
    end = '2024-01-01'

    # Downloading data
    data = yf.download(assets, start=start, end=end)['Adj Close']
    data.columns = assets

    print("Data shape:", data.shape)
    print("Data index type:", type(data.index))
    print("First 5 rows of data:\n", data.head())
    return data


def rebalance_portfolio(data, start, end):
    # Rebalancing periods
    rebalancing_periods = [3, 6, 12]  # months

    # Parameter grid
    param_grid = {
        'model': ['HRP'],
        'codependence': ['pearson', 'spearman', 'abs_pearson', 'abs_spearman'],
        'rm': ['MV', 'MAD', 'MSV'],
        'linkage': ['single', 'complete', 'average', 'ward'],
        'max_k': [10], # Higher values (15, 20, 25, 30) might not provide additional benefit, as they exceed the number of assets.
        'leaf_order': [True, False]
    }

    # Generate all combinations
    all_combinations = list(itertools.product(*param_grid.values()))

    # Store results
    results = {}

    for months in rebalancing_periods:
        Y = data[assets].resample(f'{months}M').last().pct_change().dropna()

        print(f"Rebalancing period: {months} months")
        print("Y shape:", Y.shape)

        for combo in all_combinations:
            params = dict(zip(param_grid.keys(), combo))
            port = rp.HCPortfolio(returns=Y)

            # Portfolio optimization
            w = port.optimization(model=params['model'],
                                  codependence=params['codependence'],
                                  rm=params['rm'],
                                  rf=0,  # Risk free rate
                                  linkage=params['linkage'],
                                  max_k=params['max_k'],
                                  leaf_order=params['leaf_order'])

            # Debugging: Print w and its sum
            print("Weights (w):\n", w)
            print("Sum of weights:", w['weights'].sum())

            # Check if weights are valid
            if w.isnull().any().any() or not np.isclose(w['weights'].sum(), 1, atol=0.0001):
                print("Invalid weights detected, skipping...")
                continue

            # Backtesting
            port_returns = (Y * w['weights']).sum(axis=1)
            sharpe_ratio = port_returns.mean() / port_returns.std() * (12 / months)**0.5
            key = (months, *combo)
            results[key] = {'Return': port_returns.mean() * 12, 'Volatility': port_returns.std() * (12 / months)**0.5, 'Sharpe Ratio': sharpe_ratio}

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    # Find the best parameter set based on the highest Sharpe Ratio
    best_key = max(results, key=lambda x: results[x]['Sharpe Ratio'])
    best_params = results[best_key]
    print("Best Parameters:", best_key)
    print("Performance Metrics:", best_params)

    # Download S&P 500 data for the same period for benchmarking
    sp500 = yf.download('^GSPC', start=start, end=end)['Adj Close'].pct_change().dropna()

    # Apply the best parameters to the full dataset
    port = rp.HCPortfolio(returns=data[assets].pct_change().dropna())
    w = port.optimization(model=best_key[1],
                          codependence=best_key[2],
                          rm=best_key[3],
                          rf=0,
                          linkage=best_key[4],
                          max_k=best_key[5],
                          leaf_order=best_key[6])

    # Calculate portfolio returns
    portfolio_returns = (data[assets].pct_change().dropna() * w['weights']).sum(axis=1)
    print("Portfolio returns shape:", portfolio_returns.shape)

    # Ensure the index is a datetime index for the portfolio returns
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    # Cumulative returns for the portfolio
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Ensure the index is a datetime index for the S&P 500
    sp500.index = pd.to_datetime(sp500.index)

    # Cumulative returns for the S&P 500
    cumulative_sp500 = (1 + sp500).cumprod()

    return cumulative_returns, cumulative_sp500


def plot_hrp_rebalancing(cumulative_returns, cumulative_sp500):
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(cumulative_returns.index, cumulative_returns, label='HRP Portfolio')
    plt.plot(cumulative_sp500.index, cumulative_sp500, label='S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns of HRP Portfolio vs S&P 500')
    plt.legend()
    plt.show()


def run_hrp_rebalancing(start, end):
    # start = '2018-01-01'
    # end = '2024-01-01'
    data = download_data(start, end)
    cumulative_returns, cumulative_sp500 = rebalance_portfolio(data, start, end)
    plot_hrp_rebalancing(cumulative_returns, cumulative_sp500)


if __name__ == "__main__":

    run_hrp_rebalancing('2020-01-01', '2024-01-01')

