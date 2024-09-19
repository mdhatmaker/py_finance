import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm


# https://medium.com/@serdarilarslan/value-at-risk-var-and-its-implementation-in-python-5c9150f73b0e
"""
1. What is Value at Risk (VaR)?
Value at Risk (VaR) quantifies the potential loss in value of a portfolio over a defined period for a given confidence interval.
For instance, a one-day VaR at a 95% confidence level indicates that there is a 95% chance that the portfolio will not lose more
than the VaR amount in one day.
2. Why is VaR Important?
VaR is a critical tool in risk management for financial institutions and investors because it:
Provides a clear metric for potential losses.
Helps in setting risk limits and capital reserves.
Aids in regulatory compliance and risk reporting.
Facilitates communication of risk to stakeholders.
"""


# Value at Risk - historical
def get_VaR_historical(symbol: str, startDate: str="2020-01-01", endDate=None, confidence_level=0.95):
    if not endDate:
        endDate = datetime.now()

    # Fetch historical data for a stock
    data = yf.download(symbol, start=startDate, end=endDate)
    returns = data['Adj Close'].pct_change().dropna()

    # Calculate the historical VaR at 95% confidence level
    # confidence_level = 0.95
    VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)

    print(f"Historical VaR (95% confidence level): {VaR_historical:.2%}")

    return returns, VaR_historical


def plot_VaR_historical(returns, VaR_historical):
    # Plot the historical returns and VaR threshold
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(VaR_historical, color='red', linestyle='--', label=f'VaR (95%): {VaR_historical:.2%}')
    plt.title('Historical Returns of AAPL')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# Value at Risk - variance-covariance
def get_VaR_variance_covariance(symbol: str, startDate: str="2020-01-01", endDate=None, confidence_level=0.95):
    if not endDate:
        endDate = datetime.now()

    # Fetch historical data for a stock
    data = yf.download(symbol, start=startDate, end=endDate)
    returns = data['Adj Close'].pct_change().dropna()

    # Calculate the mean and standard deviation of returns
    mean_return = np.mean(returns)
    std_dev = np.std(returns)

    # Calculate the VaR at 95% confidence level using the Z-score
    # confidence_level = 0.95
    z_score = norm.ppf(1 - confidence_level)
    VaR_variance_covariance = mean_return + z_score * std_dev

    print(f"Variance-Covariance VaR (95% confidence level): {VaR_variance_covariance:.2%}")

    return mean_return, std_dev, VaR_variance_covariance


def plot_VaR_variance_covariance(mean_return, std_dev, VaR_variance_covariance):
    # Plot the normal distribution and VaR threshold
    plt.figure(figsize=(10, 6))
    x = np.linspace(mean_return - 3*std_dev, mean_return + 3*std_dev, 1000)
    y = norm.pdf(x, mean_return, std_dev)
    plt.plot(x, y, label='Normal Distribution')
    plt.axvline(VaR_variance_covariance, color='red', linestyle='--', label=f'VaR (95%): {VaR_variance_covariance:.2%}')
    plt.fill_between(x, 0, y, where=(x <= VaR_variance_covariance), color='red', alpha=0.5)
    plt.title('Normal Distribution of Returns with VaR Threshold')
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


# Value at Risk - monte carlo
def get_VaR_monte_carlo(symbol: str, startDate: str="2020-01-01", endDate=None, confidence_level=0.95):
    if not endDate:
        endDate = datetime.now()

    # Fetch historical data for a stock
    data = yf.download(symbol, start=startDate, end=endDate)
    returns = data['Adj Close'].pct_change().dropna()

    # Simulate future returns using Monte Carlo
    num_simulations = 10000
    simulation_horizon = 252  # Number of trading days in a year
    simulated_returns = np.random.normal(np.mean(returns), np.std(returns), (simulation_horizon, num_simulations))

    # Calculate the simulated portfolio values
    initial_investment = 1000000  # $1,000,000
    portfolio_values = initial_investment * np.exp(np.cumsum(simulated_returns, axis=0))

    # Calculate the portfolio returns
    portfolio_returns = portfolio_values[-1] / portfolio_values[0] - 1

    # Calculate the VaR at 95% confidence level
    # confidence_level = 0.95
    VaR_monte_carlo = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    print(f"Monte Carlo VaR (95% confidence level): {VaR_monte_carlo:.2%}")

    return portfolio_returns, VaR_monte_carlo


def plot_VaR_monte_carlo(portfolio_returns, VaR_monte_carlo):
    # Plot the distribution of simulated portfolio returns and VaR threshold
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(VaR_monte_carlo, color='red', linestyle='--', label=f'VaR (95%): {VaR_monte_carlo:.2%}')
    plt.title('Simulated Portfolio Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def run_VaR_historical(ticker: str, startDate: str, endDate: str = None):
    returns, VaR_historical = get_VaR_historical(ticker, startDate, endDate)
    plot_VaR_historical(returns, VaR_historical)

    mean_return, std_dev, VaR_variance_covariance = get_VaR_variance_covariance(ticker, startDate, endDate)
    plot_VaR_variance_covariance(mean_return, std_dev, VaR_variance_covariance)

    portfolio_returns, VaR_monte_carlo = get_VaR_monte_carlo(ticker, startDate, endDate)
    plot_VaR_monte_carlo(portfolio_returns, VaR_monte_carlo)


if __name__ == '__main__':

    symbol = 'AMD'
    startDate = '2022-04-01'
    endDate = None

    run_VaR_historical(ticker=symbol, startDate='2022-04-01', endDate=None)

