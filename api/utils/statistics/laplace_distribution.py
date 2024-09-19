import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# https://www.askpython.com/python/examples/laplace-distribution-python


# Define location (mean) and scale parameters
loc = 2.5
scale = 1.0

# Create a Laplace distribution object
laplace_dist = stats.laplace(loc, scale)

# Calculate the PDF for a specific value (x = 3.0)
x = 3.0
pdf_value = laplace_dist.pdf(x)

print(f"PDF for x = {x}: {pdf_value}")

# Alternatively, calculate the PDF for an array of values
x_values = np.linspace(1.0, 4.0, 10)
pdf_values = laplace_dist.pdf(x_values)

# Plot the PDF (using matplotlib)
import matplotlib.pyplot as plt

plt.plot(x_values, pdf_values)
plt.xlabel("x")
plt.ylabel("PDF(x)")
plt.title("Laplace Distribution PDF")
plt.show()




# Simulate daily stock returns
num_days = 250
drift = 0.001  # Daily average return
volatility = 0.01  # Daily volatility
scale = volatility / np.sqrt(2)  # Scale parameter for Laplace distribution

# Generate random Laplace distributed returns
returns = drift + scale * np.random.laplace(loc=0, scale=1, size=num_days)

# Calculate daily stock prices (assuming starting price of $100)
prices = np.cumsum(returns) + 100

# Simulate a market crash by randomly dropping the price on a specific day
crash_day = np.random.randint(1, num_days)
crash_severity = -0.2  # Proportionate price drop during crash
prices[crash_day] *= (1 - crash_severity)

# Plot the stock prices
plt.plot(prices)
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.title("Stock Price Simulation with Laplace Distribution (Crash on Day {})".format(crash_day))
plt.grid(True)
plt.show()

# Analyze daily returns with Laplace distribution
laplace_dist = stats.laplace(loc=drift, scale=scale)

# Calculate theoretical percentiles for returns (e.g., 1st, 5th, 95th, 99th)
percentiles = np.array([1, 5, 95, 99])
theoretical_percentiles = laplace_dist.ppf(percentiles / 100)

# Calculate actual percentiles from simulated returns
actual_percentiles = np.percentile(returns, percentiles)

# Print the comparison of theoretical vs. actual percentiles
print("Theoretical Percentiles (Daily Returns):")
print(theoretical_percentiles)
print("\nActual Percentiles (Simulated Returns):")
print(actual_percentiles)




# Define parameters for the Laplace distribution (hydrology scenario)
loc = 1.5  # Average daily rainfall (mm)
scale = 0.8  # Controls spread (larger = more frequent extreme events)

# Simulate 365 days of rainfall data
daily_rainfall = np.random.laplace(loc, scale, size=365)

# Ensure non-negative rainfall values
daily_rainfall[daily_rainfall < 0] = 0  # Set negative values to zero

# Plot the daily rainfall data
days = np.arange(len(daily_rainfall))  # Days (x-axis)

plt.figure(figsize=(10, 6))
plt.plot(days, daily_rainfall, marker='o', linestyle='-', label='Daily Rainfall (mm)')

# Optional: Add labels and title
plt.xlabel('Day')
plt.ylabel('Rainfall (mm)')
plt.title('Simulated Daily Rainfall (Laplace Distribution)')
plt.grid(True)
plt.legend()

plt.show()

# Print some descriptive statistics (optional)
print(f"Average daily rainfall: {daily_rainfall.mean():.2f} mm")
print(f"Maximum daily rainfall: {daily_rainfall.max():.2f} mm")




"""
Laplace distribution is a probability distribution used to model data with heavy tails, where extreme
values are more likely than in a normal distribution. It has wider tails than the normal distribution,
making it suitable for modeling financial asset returns, rainfall patterns, and other phenomena with
occasional extreme values. The key parameters are the location (mean) and scale (related to spread
or volatility).

Laplace distribution is used in multiple fields, from Physics to Engineering. It is very similar to a
normal distribution, but it has much wider tails, so it can model instances where the deviations are
very large.
"""


