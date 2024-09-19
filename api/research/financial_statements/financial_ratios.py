import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/financial-ratio-analysis-python


def get_income_statement(ticker):
    # Fetch the stock's financial data
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials.T
    return income_stmt


def plot_gross_profit_margin(ticker, income_stmt, nyears):
    # Keep only the last nyears years of data
    income_stmt = income_stmt.tail(nyears)
    # Calculate Gross Profit Margin
    income_stmt['Gross Profit Margin'] = (income_stmt['Gross Profit'] / income_stmt['Total Revenue']) * 100
    # Display the Gross Profit Margin
    print(f"Gross Profit Margin for {ticker} over the last {nyears} years:")
    print(income_stmt['Gross Profit Margin'])
    # Plot the Gross Profit Margin
    plt.figure(figsize=(10, 6))
    plt.plot(income_stmt.index, income_stmt['Gross Profit Margin'], marker='o', linestyle='-', color='b')
    plt.title(f'Gross Profit Margin of {ticker} Over the Last {nyears} Years')
    plt.xlabel('Year')
    plt.ylabel('Gross Profit Margin (%)')
    plt.grid(True)
    plt.show()


def plot_operating_profit_margin(ticker, income_stmt, nyears):
    # Filter data for the last nyears years
    years = income_stmt.index[:nyears]
    # Calculate the operating profit margin
    income_stmt['Operating Profit Margin'] = (income_stmt['Operating Income'] / income_stmt['Total Revenue']) * 100
    # Filter the data for the last nyears years
    op_margin_years = income_stmt.loc[years, 'Operating Profit Margin']
    # Plot the operating profit margin
    plt.figure(figsize=(10, 6))
    plt.plot(op_margin_years.index, op_margin_years.values, marker='o', linestyle='-', color='b')
    plt.title(f'Operating Profit Margin of {ticker} - Last {nyears} Years')
    plt.xlabel('Year')
    plt.ylabel('Operating Profit Margin (%)')
    plt.grid(True)
    plt.show()
    # Print the operating profit margins
    print(f"Operating Profit Margin for {ticker} - Last {nyears} Years")
    print(op_margin_years)


def plot_net_profit_margin(ticker, income_statement, nyears):
    # Calculate net profit margin
    net_income = income_statement['Net Income']
    total_revenue = income_statement['Total Revenue']
    net_profit_margin = (net_income / total_revenue) * 100
    # Plot the net profit margin as a line chart
    plt.figure(figsize=(10, 6))
    net_profit_margin.plot(marker='o', color='b', linestyle='-')
    plt.title(f'Net Profit Margin {ticker}')
    plt.xlabel('Year')
    plt.ylabel('Net Profit Margin (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Display the net profit margin for the last five years
    print(f"Net Profit Margin of {ticker} for the Last {nyears} Years:")
    print(net_profit_margin.tail(nyears))


def run_financial_ratios(ticker, nyears=4):
    income_stmt = get_income_statement(ticker)
    plot_gross_profit_margin(ticker, income_stmt, nyears)
    plot_operating_profit_margin(ticker, income_stmt, nyears)
    plot_net_profit_margin(ticker, income_stmt, nyears)


if __name__ == "__main__":

    run_financial_ratios('AAPL', 4)

