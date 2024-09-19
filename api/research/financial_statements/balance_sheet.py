import yfinance as yf
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/python-balance-sheet-analysis


def get_income_statement(ticker):
    # Fetch income statement data from Yahoo Finance
    stock = yf.Ticker(ticker)
    income_statement = stock.financials.loc['Net Income']
    return income_statement


def get_assets(ticker):
    # Fetch balance sheet data from Yahoo Finance
    stock = yf.Ticker(ticker)
    assets = stock.balance_sheet.loc['Total Assets']
    return assets


def analyze_income_statement(income_statement):
    # Calculate average net income over the last 5 years
    avg_net_income = income_statement.tail(5).mean()
    return avg_net_income


def analyze_assets(assets):
    # Calculate average assets over the last 5 years
    avg_assets = assets.tail(5).mean()
    return avg_assets


def plot_financials(income_statement, assets):
    # Plotting income statement and assets data
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Net Income (in Billions)', color=color)
    income_statement.tail(5).plot(kind='bar', color=color, ax=ax1, position=1, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(income_statement.tail(5).index.strftime('%Y'), rotation=45)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Total Assets (in Billions)', color=color)  # we already handled the x-label with ax1
    assets.tail(5).plot(kind='bar', color=color, ax=ax2, position=0, width=0.4)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Net Income and Total Assets for Apple over the Last 5 Years')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def run_balance_sheet(ticker_symbol: str):
    # Fetch income statement and assets data
    # ticker_symbol = "AAPL"
    income_statement = get_income_statement(ticker_symbol)
    assets = get_assets(ticker_symbol)
    # Analyze income statement and assets
    avg_net_income = analyze_income_statement(income_statement)
    avg_assets = analyze_assets(assets)
    print("Average Net Income for Apple over the last 5 years:", avg_net_income)
    print("Average Total Assets for Apple over the last 5 years:", avg_assets)
    # Plot income statement and assets
    plot_financials(income_statement, assets)


if __name__ == "__main__":

    run_balance_sheet("AAPL")

