import numpy as np
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/survival-analysis-python


# --- Generate Random Survival Data with NumPy ---

# Set random seed (optional, for reproducibility)
np.random.seed(42)

# Simulate random durations (time) between 0 and 10
durations = np.random.uniform(low=0, high=10, size=100)

# Simulate random events (0 for no event, 1 for event)
# Adjust probability (p) to control the number of events
events = np.random.binomial(n=1, p=0.3, size=100)  # Assuming 30% event probability

# Sort together by duration (ascending)
data = np.array([durations, events]).T
data = data[data[:, 0].argsort()]

# Separate sorted data
sorted_durations = data[:, 0]
sorted_events = data[:, 1]

# Initialize variables for Kaplan-Meier calculation
n_total = len(sorted_durations)
n_alive = n_total
t = []
s = []  # Survival probability

for i in range(n_total):
    if sorted_events[i] == 1:
        t.append(sorted_durations[i])
        s.append(n_alive / (n_alive + 1))
    n_alive -= 1

# --- Plot Kaplan-Meier Curve (DIY) ---

plt.figure(figsize=(8, 6))
plt.step(t, s, where='post')  # Step function for Kaplan-Meier curve
plt.xlabel("Time")
plt.ylabel("Probability of Survival")
plt.grid(True)
plt.title("Kaplan-Meier Survival Curve (DIY with NumPy)")
plt.show()

# --- Note on Cox Proportional Hazards Model ---

print("Cox Proportional Hazards Model cannot be directly implemented with NumPy alone.")
print("Consider using libraries like lifelines or statsmodels for this analysis.")




# Set random seed (optional, for reproducibility)
np.random.seed(42)

# Sample sizes (adjust for desired imbalance)
n_drug = 50
n_standard = 70

# Simulate durations (potentially affected by treatment)
# Drug group might have slightly longer survival times on average
drug_durations = np.random.normal(loc=5, scale=1.5, size=n_drug)
standard_durations = np.random.normal(loc=4, scale=1, size=n_standard)

# Simulate events (considering potential treatment effect)
drug_events = np.random.binomial(n=1, p=0.4, size=n_drug)  # Assuming lower event probability for drug
standard_events = np.random.binomial(n=1, p=0.6, size=n_standard)  # Assuming higher event probability for standard

# Simulate treatment assignment (0: Standard, 1: Drug)
treatment = np.concatenate((np.zeros(n_standard), np.ones(n_drug)))

# Combine data into arrays
durations = np.concatenate((standard_durations, drug_durations))
events = np.concatenate((standard_events, drug_events))
sorted_data = np.array([durations, events, treatment]).T  # Combine and sort by duration
sorted_data = sorted_data[sorted_data[:, 0].argsort()]

# Separate sorted data
sorted_durations = sorted_data[:, 0]
sorted_events = sorted_data[:, 1]
treatments = sorted_data[:, 2]  # Separate treatment assignments

# Initialize variables for Kaplan-Meier calculation (per treatment group)
n_total_drug = np.sum(treatments == 1)
n_total_standard = np.sum(treatments == 0)
n_alive_drug = n_total_drug
n_alive_standard = n_total_standard
t_drug = []
s_drug = []  # Survival probability (Drug)
t_standard = []
s_standard = []  # Survival probability (Standard)

current_time = sorted_durations[0]
event_index = 0

while event_index < len(sorted_events):
    if sorted_durations[event_index] > current_time:
        # Update time point
        current_time = sorted_durations[event_index]
        # Update survival probabilities for each group if applicable
        if n_alive_drug > 0:
            s_drug.append(n_alive_drug / (n_total_drug))
        if n_alive_standard > 0:
            s_standard.append(n_alive_standard / (n_total_standard))
        t_drug.append(current_time)
        t_standard.append(current_time)
    else:
        # Handle event (decrement alive for the corresponding treatment group)
        if sorted_events[event_index] == 1 and treatments[event_index] == 1:
            n_alive_drug -= 1
        elif sorted_events[event_index] == 1 and treatments[event_index] == 0:
            n_alive_standard -= 1
    event_index += 1

# Plot Kaplan-Meier curves (DIY) ---

plt.figure(figsize=(8, 6))
plt.step(t_drug, s_drug, where='post', label='New Drug')
plt.step(t_standard, s_standard, where='post', label='Standard Treatment')
plt.xlabel("Time")
plt.ylabel("Probability of Survival")
plt.grid(True)
plt.title("Kaplan-Meier Survival Curves (DIY with NumPy)")
plt.legend()
plt.show()

# --- Note on Statistical Comparison ---

print("This example demonstrates Kaplan-Meier curves. Consider tests")
print("like the Log-Rank test to statistically compare survival between groups.")






"""
Survival analysis is a statistical method used to calculate the time until an event of interest occurs.
It is applied in various fields, such as engineering and medical sciences, to evaluate the viability
of different approaches or treatments. Python provides tools like NumPy and Matplotlib to generate
random survival data, calculate survival probabilities using the Kaplan-Meier method, and visualize
survival curves for comparison.
"""


