import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/python-pl-statement-analysis


def get_revenue(ticker):
    # Fetch income statement data from Yahoo Finance
    stock = yf.Ticker(ticker)
    revenue = stock.financials.loc['Total Revenue']
    return revenue


# Return multiple years of income statements
def get_income_statements(ticker):
    # Extract financial data
    stock = yf.Ticker(ticker)
    income_statements = stock.financials.T  # Transpose to have years as rows
    return income_statements


def plot_revenue(ticker, revenue, nyears):
    # Extract revenue data
    years = revenue.tail(nyears).index.year
    revenue_values = revenue.tail(nyears).values / 1e9  # Convert to billions
    # Plotting revenue data
    plt.figure(figsize=(10, 6))
    plt.bar(years, revenue_values, color='skyblue')
    plt.title(f'Total Revenue for {ticker} over the Last {nyears} Years')
    plt.xlabel('Year')
    plt.ylabel('Total Revenue (in Billions)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_profit_by_year(ticker, income_statements, nyears):
    # Extract the years' profits (net income)
    profits = income_statements['Net Income'].iloc[:nyears]
    # Prepare data for plotting
    years = profits.index.strftime('%Y')
    profits = profits.values
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.bar(years, profits, color='blue')
    plt.xlabel('Year')
    plt.ylabel('Net Income (USD)')
    plt.title(f'{ticker} Net Income (Profits) over the Last {nyears} Years')
    plt.show()


def run_analyze_financial_statements(ticker_symbol, nyears=4):
    # Plot revenue for past nyears years
    revenue = get_revenue(ticker_symbol)
    plot_revenue(ticker_symbol, revenue, nyears)

    # Plot profits for past nyears years
    income_statements = get_income_statements(ticker_symbol)
    plot_profit_by_year(ticker_symbol, income_statements, nyears)


if __name__ == "__main__":
    # Plot revenue and profit data for Apple (AAPL)
    run_analyze_financial_statements('AAPL', nyears=3)


"""
Breaking Down the P&L Statement

Revenue: Revenue is the earnings the company generates from its normal business operations by selling products or services. Steel Authority of India Ltd. (SAIL) is the revenue from stainless steel it manufactures and sells.

Expense: Expense is the opposite of revenue in some sense. It is the cost incurred by selling a product or service. For example, SAIL has expenses such as raw materials, distribution charges, etc.

Profit Before Tax (PBT): Profit is defined as Expenses subtracted from Revenue. PBT is the same: the average profit from the company’s everyday operations.

Profit After Tax (PAT): Profit After Tax is PBT, from which the tax portion is subtracted. For example, the profit is Rs 100 (PBT), the interest is Rs 20, and PAT is { Rs 100 – Rs 20 }. Every country has different taxes, and this component is not uniform across all regions. This is also referred to as Net Income.

Earnings per Share (EPS): Earnings Per Share (EPS) represents a company’s profitability on a per-share basis, offering a direct insight into financial health. They are calculated by dividing Net Income by outstanding shares issued by the company.
"""