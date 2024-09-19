import numpy as np
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/bayesian-inference-in-python


# Generate some synthetic data
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, size=100)

# Define the prior hyperparameters
prior_mu_mean = 0
prior_mu_precision = 1  # Variance = 1 / precision
prior_sigma_alpha = 2
prior_sigma_beta = 2  # Beta = alpha / beta

# Update the prior hyperparameters with the data
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma ** 2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data)) ** 2) / 2

# Calculate the posterior parameters
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plot the posterior distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior distribution of $\mu$')
plt.xlabel('$\mu$')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior distribution of $\sigma$')
plt.xlabel('$\sigma$')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Calculate summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu:", mean_mu)
print("Standard deviation of mu:", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma:", mean_sigma)
print("Standard deviation of sigma:", std_sigma)

# Observed data
num_visitors = 1000  # Total number of visitors to the website
num_conversions = 50  # Number of conversions (desired actions)

# Prior hyperparameters for the Beta distribution
prior_alpha = 1  # Shape parameter
prior_beta = 1  # Shape parameter

# Update the prior with the observed data to get the posterior parameters
posterior_alpha = prior_alpha + num_conversions
posterior_beta = prior_beta + (num_visitors - num_conversions)

# Generate samples from the posterior Beta distribution
posterior_samples = np.random.beta(posterior_alpha, posterior_beta, size=10000)

# Plot the posterior distribution
plt.figure(figsize=(8, 6))
plt.hist(posterior_samples, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Posterior Distribution of Conversion Rate')
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.xlim(0, 0.1)  # Limiting x-axis to focus on conversion rates close to zero
plt.show()

# Calculate summary statistics
mean_conversion_rate = posterior_alpha / (posterior_alpha + posterior_beta)
mode_conversion_rate = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)  # Mode of the Beta distribution

print("Mean conversion rate:", mean_conversion_rate)
print("Mode conversion rate:", mode_conversion_rate)




"""
Bayesian inference is a statistical method based on Bayes’s theorem, which updates the probability of
an event as new data becomes available. It is widely used in various fields, such as finance, medicine,
and engineering, to make predictions and decisions based on prior knowledge and observed data.
In Python, Bayesian inference can be implemented using libraries like NumPy and Matplotlib to generate
and visualize posterior distributions.
"""





