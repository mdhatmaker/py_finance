import yfinance as yf
import pandas as pd
from api.utils.yahoo_finance import get_income_statement

# https://www.askpython.com/python/fundamental-financial-analysis-python


# Pass an Excel file that has been downloaded from Screener
def analyze_income_statement_excel(excel_filename, sheet_name):
    # excel_filename = 'your_file.xlsx'
    try:
        # Replace 'your_file.xlsx' with the actual path to your Excel file
        data = pd.read_excel(excel_filename)

        # Print the first 5 rows of the data (assuming you have data)
        print(data.head())

        # Access a specific sheet, handling potential errors gracefully
        try:
            sheet_name = 'Cash Flow'  # Replace with your desired sheet name
            data_sheet = pd.read_excel(excel_filename, sheet_name=sheet_name)
            print(f"\nData from sheet '{sheet_name}':")
            print(data_sheet.head())
        except Exception as e:
            print(f"\nError accessing sheet '{sheet_name}': {e}")

    except Exception as e:
        print(f"Error reading Excel file: {e}")


def analyze_income_statement_yfinance(income_statement):
    # Calculate metrics or perform analysis
    # Example: calculate average net income over the last 5 years
    avg_net_income = income_statement.mean()
    return avg_net_income


def run_analyze_income_statement(ticker_symbol: str):
    income_statement = get_income_statement(ticker_symbol)
    avg_net_income = analyze_income_statement_yfinance(income_statement)
    print("Average Net Income:", avg_net_income)


if __name__ == "__main__":

    # Example usage of getting financial data from Yahoo Finance
    run_analyze_income_statement("AAPL")

