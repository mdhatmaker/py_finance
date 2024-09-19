# Importing required libraries
from scipy.stats import uniform
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, binom, bernoulli, poisson, expon, norm

# https://www.askpython.com/python/examples/probability-distributions


# taking random variables from Uniform distribution
data = uniform.rvs(size=100000, loc=5, scale=10)

# Plotting the results
sb.set_style('whitegrid')
ax = sb.distplot(data, bins=30, color='k')
ax.set(xlabel='interval')
plt.show()



# Applying the binom class
pb = binom(n=20, p=0.6)

x = np.arange(1, 21)
pmf = pb.pmf(x)

# Visualizing the distribution
sb.set_style('whitegrid')
plt.vlines(x, 0, pb.pmf(x), colors='k', linestyles='-', lw=5)
plt.ylabel('Probability')
plt.xlabel('Intervals')
plt.show()



# Applying the bernoulli class
data = bernoulli.rvs(size=1000, p=0.8)

# Visualizing the results
sb.set_style('whitegrid')
sb.displot(data, discrete=True, shrink=.8, color='k')
plt.show()



# Applying the poisson class methods
x = np.arange(0, 10)
pmf = poisson.pmf(x, 3)

# Visualizing the results
sb.set_style('whitegrid')
plt.vlines(x, 0, pmf, colors='k', linestyles='-', lw=6)
plt.ylabel('Probability')
plt.xlabel('intervals')
plt.show()



# Applying the expon class methods
x = np.linspace(0.001, 10, 100)
pdf = expon.pdf(x)

# Visualizing the results
sb.set_style('whitegrid')
plt.plot(x, pdf, 'r-', lw=2, alpha=0.6, label='expon pdf', color='k')
plt.xlabel('intervals')
plt.ylabel('Probability Density')
plt.show()



# Creating the distribution
data = np.arange(1, 10, 0.01)
pdf = norm.pdf(data, loc=5.3, scale=1)

# Visualizing the distribution

sb.set_style('whitegrid')
sb.lineplot(data, pdf, color='black')
plt.ylabel('Probability Density')


