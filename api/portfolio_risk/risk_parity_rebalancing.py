import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# https://medium.com/@crisvelasquez/optimize-portfolio-performance-with-risk-parity-rebalancing-in-python-b9931d9e785b


# Tickers of top companies from S&P 500
tickers = ['AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'BRK-B', 'NVDA', 'JPM',
           'JNJ', 'V', 'PG', 'UNH', 'MA', 'DIS', 'HD', 'PYPL', 'BAC', 'CMCSA']
tickers.sort()


def fetch_data(start, end):
    data = yf.download(tickers + ['^GSPC'], start, end)['Adj Close']
    return data


def fetch_returns(start, end):
    data = yf.download(tickers + ['^GSPC'], start, end)['Adj Close']
    return data.pct_change().dropna()


def calculate_weights(data):
    returns = data.pct_change().dropna()
    vol = data.rolling(window=60).std().dropna().iloc[-1][:-1]  # Exclude S&P 500 index
    inv_vol = 1 / vol
    weights = inv_vol / np.sum(inv_vol)
    return weights * 100    # Convert to percentage


def simulate_portfolio(returns, weights, n_days=60):
    port_val = [1]
    sp500_val = [1]
    weights = np.ones(len(tickers)) / len(tickers)  # Start with equal weights

    for i in range(len(returns)):
        if i < 60:  # If less than rolling window, use equal weights
            daily_port_return = np.dot(returns.iloc[i][:-1], weights)
        else:
            if i % n_days == 0:  # Rebalancing
                weights = calculate_weights(returns.iloc[i-60:i])
            daily_port_return = np.dot(returns.iloc[i][:-1], weights)

        port_val.append(port_val[-1] * (1 + daily_port_return))
        sp500_val.append(sp500_val[-1] * (1 + returns.iloc[i]['^GSPC']))
    return port_val, sp500_val


def plot_weights(weights):
    plt.figure(figsize=(12, 6))
    weights_sorted = weights.sort_values()
    ax = weights_sorted.plot(kind='barh', color='skyblue')
    plt.title('Risk Parity Weights (%)')
    plt.xlabel('Weights (%)')
    plt.ylabel('Tickers')

    # Adding labels to the bars
    for i, v in enumerate(weights_sorted):
        ax.text(v, i, f"{v:.2f}%", va='center', fontweight='light', fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_risk_parity_rebalancing(returns, port_val, sp500_val):
    plt.figure(figsize=(14, 7))
    plt.plot(returns.index, port_val[:-1], label='Risk Parity Portfolio')
    plt.plot(returns.index, sp500_val[:-1], label='S&P 500', alpha=0.6)
    plt.legend()
    plt.title('Risk Parity vs. S&P 500')

    # Annotations for initial and final values
    initial_val = 10000
    final_rp = port_val[-2] * initial_val  # port_val[-2] because we have an extra entry in port_val list
    final_sp500 = sp500_val[-2] * initial_val  # same reason here

    plt.annotate(f"${initial_val:.2f}", (returns.index[0], port_val[0]),
                 xytext=(-60,0), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"), fontsize=15)
    plt.annotate(f"${final_rp:.2f}", (returns.index[-1], port_val[-2]),
                 xytext=(15,15), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"), fontsize=15)
    plt.annotate(f"${initial_val:.2f}", (returns.index[0], sp500_val[0]),
                 xytext=(-60,-20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"), fontsize=15)
    plt.annotate(f"${final_sp500:.2f}", (returns.index[-1], sp500_val[-2]),
                 xytext=(15,-15), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->"), fontsize=15)

    plt.show()


def calculate_rolling_metrics(returns, port_val):
    # Calculate rolling metrics
    rolling_vol_rp = pd.Series(port_val[:-1]).pct_change().rolling(window=60).std()
    rolling_vol_sp = returns['^GSPC'].rolling(window=60).std()
    rolling_sharpe_rp = pd.Series(port_val[:-1]).pct_change().rolling(window=60).mean() / rolling_vol_rp
    rolling_sharpe_sp = returns['^GSPC'].rolling(window=60).mean() / rolling_vol_sp
    # Calculate weights over time
    weights_df = pd.DataFrame(index=returns.index, columns=tickers)
    for i, date in enumerate(returns.index):
        if i < 60:
            weights_df.loc[date] = np.ones(len(tickers)) / len(tickers)
        elif i % 60 == 0:
            weights_df.loc[date] = calculate_weights(returns.iloc[i-60:i])
        else:
            weights_df.loc[date] = weights_df.iloc[i-1]
    return weights_df, rolling_vol_rp, rolling_vol_sp, rolling_sharpe_rp, rolling_sharpe_sp


def plot_rolling_metrics(returns, weights_df, rolling_vol_rp, rolling_vol_sp, rolling_sharpe_rp, rolling_sharpe_sp):
    # Create subplots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,2])
    # Plot rolling volatility
    ax0 = plt.subplot(gs[0])
    ax0.plot(returns.index, rolling_vol_rp, label='Risk Parity Rolling Volatility')
    ax0.plot(returns.index, rolling_vol_sp, label='S&P 500 Rolling Volatility', alpha=0.6)
    ax0.legend()
    ax0.set_title('Rolling Volatility')
    # Plot rolling Sharpe ratio
    ax1 = plt.subplot(gs[1])
    ax1.plot(returns.index, rolling_sharpe_rp, label='Risk Parity Rolling Sharpe Ratio')
    ax1.plot(returns.index, rolling_sharpe_sp, label='S&P 500 Rolling Sharpe Ratio', alpha=0.6)
    ax1.legend()
    ax1.set_title('Rolling Sharpe Ratio')
    # Plot asset weights over time
    ax2 = plt.subplot(gs[2])
    weights_df.plot(kind='area', stacked=True, ax=ax2, legend=False)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title('Asset Weights Over Time')
    # Show plot
    plt.tight_layout()
    plt.show()


def run_parity_rebalancing(start, end, n_days=60):
    # start = "2010-01-01"
    # end = "2023-01-01"
    data = fetch_data(start, end)
    # Portfolio weights
    weights = calculate_weights(data)
    plot_weights(weights)
    # Performance evaluation
    returns = fetch_returns(start, end)
    port_val, sp500_val = simulate_portfolio(returns, weights, n_days)
    plot_risk_parity_rebalancing(returns, port_val, sp500_val)
    # Rolling metrics
    weights_df, rolling_vol_rp, rolling_vol_sp, rolling_sharpe_rp, rolling_sharpe_sp = calculate_rolling_metrics(returns, port_val)
    plot_rolling_metrics(returns, weights_df, rolling_vol_rp, rolling_vol_sp, rolling_sharpe_rp, rolling_sharpe_sp)


if __name__ == "__main__":

    run_parity_rebalancing("2010-01-01", "2023-01-01", n_days=30)



"""
By analyzing these metrics, one can discern that for roughly the same amount of risk, the Risk Parity strategy offers twice as much returns compared to the conventional benchmark! The visual contrast between the portfolio’s rolling volatility and the S&P 500, coupled with the Sharpe ratio, offers an illuminating perspective on the quality and stability of returns.
4. Risk Parity in Diverse Market Conditions
Risk Parity stands out for its risk-balancing emphasis, especially when assessing its performance across market conditions:
4.1. Bull Markets
During bullish periods, equity-centric portfolios naturally shine, with stocks generally offering higher returns. However, Risk Parity, while not overly aggressive, captures a share of this growth. Its risk-focused approach means it might not always maximize on all bullish opportunities, but its strength lies in a more controlled participation.
4.2. Bear Markets
In downturns, the protective nature of Risk Parity becomes evident. With its adaptive rebalancing, the strategy can move away from plummeting assets, favoring safer havens. This can translate to lower drawdowns compared to traditional portfolios.
4.3. Economic Shifts
In both inflationary and deflationary times, Risk Parity’s dynamic allocation adapts to capitalize on the most favorable asset classes.
4.4. Global Impacts
Given global interconnectedness, Risk Parity’s diversification minimizes the effects of major worldwide events.
5. Criticisms and Concerns
As with any investment strategy, Risk Parity is not without its critics. Here, we lay out some of the concerns and counterarguments regarding the approach.
5.1. Dependence on Leverage
Risk Parity often involves using leverage, especially when bonds or other low-volatility assets dominate the portfolio. Critics argue that leverage can amplify losses in turbulent times, potentially negating the benefits of risk balancing.
5.2. Complexity and Costs
The dynamic nature of Risk Parity can lead to frequent rebalancing, which might result in higher transaction costs. Additionally, understanding the nuances of the strategy might be challenging for retail investors without a financial background.
5.3. Over-reliance on Quantitative Models
While quantitative models form the backbone of the Risk Parity strategy, relying solely on them can be risky. Models are as good as the assumptions they’re based on. Unprecedented market events can throw a spanner in the works, leading to unexpected results.
6. Future Prospects for Risk Parity
The financial world is ever-evolving, and strategies must adapt or risk becoming obsolete. Here’s a peek into the future of Risk Parity:
6.1. Incorporation of Alternative Assets
To diversify risk further, there’s an increasing trend toward including alternative assets like real estate, commodities, or even cryptocurrencies in Risk Parity portfolios.
6.2. Advent of AI and Machine Learning
Advanced algorithms can offer predictive insights, potentially refining the Risk Parity strategy. By forecasting macroeconomic shifts or understanding intricate asset correlations better, AI and machine learning might drive the next evolution in Risk Parity investing.
6.3. Broader Adoption by Retail Investors
With the advent of financial technology and easy-to-use investment platforms, complex strategies like Risk Parity could become more accessible to the average investor.
"""