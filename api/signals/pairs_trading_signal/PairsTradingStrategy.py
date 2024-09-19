import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Execute trades based on trading signals

class PairsTradingStrategy:
    def __init__(self, data, asset1, asset2):
        self.data = data
        self.asset1 = asset1
        self.asset2 = asset2
        self.beta = None

    def fit(self):
        X = sm.add_constant(self.data[self.asset1])
        y = self.data[self.asset2]
        model = sm.OLS(y, X).fit()
        self.beta = model.params[self.asset1]

    def generate_signals(self, zscore_threshold=1):
        spread = self.data[self.asset2] - self.beta * self.data[self.asset1]
        zscore = (spread - spread.mean()) / spread.std()
        signals = zscore > zscore_threshold
        return signals

    def execute_trades(self, signals):
        positions = []  # 1 for long, -1 for short
        for signal in signals:
            if signal:
                positions.append(1)
            else:
                positions.append(-1)

        # Execute trades based on positions
        for i in range(1, len(positions)):
            if positions[i] != positions[i-1]:
                if positions[i] == 1:
                    print(f"Buy {self.asset2} and Sell {self.asset1} at {self.data.index[i]}")
                else:
                    print(f"Sell {self.asset2} and Buy {self.asset1} at {self.data.index[i]}")


#
# # Create PairsTradingStrategy object for selected assets
# strategy = PairsTradingStrategy(data, 'AAPL', 'MSFT')
# strategy.fit()
#
# # Generate trading signals based on z-score threshold
# signals = strategy.generate_signals(zscore_threshold=1)
#
# # Execute trades based on signals
# strategy.execute_trades()

