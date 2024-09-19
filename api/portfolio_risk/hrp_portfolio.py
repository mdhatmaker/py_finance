#!pip install git+https://github.com/dcajasn/Riskfolio-Lib.git

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import riskfolio as rp

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# https://medium.com/@crisvelasquez/optimizing-portfolio-allocation-with-hierarchical-risk-parity-in-python-19b1813af618


# Tickers of top 20 companies from S&P 500 (example selection)
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'WMT', 'V', 'JPM',
          'PG', 'UNH', 'DIS', 'NVDA', 'HD', 'PYPL', 'MA', 'BAC', 'INTC', 'T']
assets.sort()


def download_data(start, end):
    # Downloading data
    data = yf.download(assets, start=start, end=end)
    # Extracting 'Adj Close' prices
    data = data['Adj Close']
    data.columns = assets
    return data


def calculate_returns(data):
    # Calculating returns
    Y = data[assets].pct_change().dropna()

    # Plotting the dendrogram for asset clusters
    ax = rp.plot_dendrogram(returns=Y,
                            codependence='pearson',
                            linkage='single',
                            k=None,
                            max_k=10,
                            leaf_order=True,
                            ax=None)

    # Building the portfolio object
    port = rp.HCPortfolio(returns=Y)

    # Parameters for the optimization
    model = 'HRP' # HRP model
    codependence = 'pearson' # Pearson correlation to group assets in clusters
    rm = 'MV' # Risk measure: variance
    rf = 0 # Risk free rate
    linkage = 'single' # Linkage method for clusters
    max_k = 10 # Max number of clusters for gap statistic (not used in HRP but kept for compatibility)
    leaf_order = True # Optimal order of leaves in dendrogram

    # Portfolio optimization
    w = port.optimization(model=model,
                          codependence=codependence,
                          rm=rm,
                          rf=rf,
                          linkage=linkage,
                          max_k=max_k,
                          leaf_order=leaf_order)

    # Displaying the weights
    #display(w.T)

    return w


#########


def calculate_weights(w):
    # Identify the first column name
    first_column_name = w.columns[0]

    # Sorting weights and assets for visualization
    sorted_indices = w.sort_values(by=first_column_name, ascending=False).index
    sorted_weights = w.sort_values(by=first_column_name, ascending=False).values

    # Create a mapping of the asset names to their integer positions
    asset_mapping = {asset_name: idx for idx, asset_name in enumerate(assets)}

    # Map the sorted_indices to their integer positions
    sorted_positions = [asset_mapping[i] for i in sorted_indices]
    sorted_assets = [assets[pos] for pos in sorted_positions]

    return sorted_assets, sorted_weights


def plot_hrp_portfolio(sorted_assets, sorted_weights):
    # Plotting the ordered vertical bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_assets, sorted_weights[:, 0], color='royalblue')
    ax.set_xlabel('Weight')
    ax.set_title('Portfolio Weights by Asset')
    ax.invert_yaxis()  # Inverting y-axis to have highest weights at the top
    ax.grid(axis='x')

    # Adding the labels
    for i, (asset, weight) in enumerate(zip(sorted_assets, sorted_weights[:, 0])):
        ax.text(weight , i, f"{weight:.2f}", va='center')  # Adjust the 0.005 offset if needed

    plt.show()


def run_hrp_portfolio(start, end):
    # start = '2020-01-01'
    # end = '2024-01-01'
    data = download_data(start, end)
    w = calculate_returns(data)
    sorted_assets, sorted_weights = calculate_weights(w)
    plot_hrp_portfolio(sorted_assets, sorted_weights)


if __name__ == "__main__":

    run_hrp_portfolio('2020-01-01', '2024-01-01')

