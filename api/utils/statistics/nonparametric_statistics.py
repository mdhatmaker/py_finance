import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# https://www.askpython.com/python/examples/non-parametric-statistics-in-python


# Sample data (replace with your actual data)
data = [2, 5, 7, 8, 2, 1, 9, 4, 5, 3, 7, 8, 2, 6, 1]

# Create the histogram
plt.hist(data, bins=10, edgecolor='black')  # Adjust 'bins' for different bin counts

# Customize the plot (optional)
plt.xlabel('Data Values')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Data')
plt.grid(True)

# Display the plot
plt.show()



# Sample data (replace with your actual data)
data = [2, 5, 7, 8, 2, 1, 9, 4, 5, 3, 7, 8, 2, 6, 1]

# Create the KDE plot
sns.kdeplot(data)

# Customize the plot (optional)
plt.xlabel('Data Values')
plt.ylabel('Probability Density')
plt.title('KDE Plot of Sample Data')
plt.grid(True)

# Display the plot
plt.show()



# Sample data sets (replace with your actual data)
data1 = [2, 5, 7, 8, 2, 1, 9, 4, 5, 3, 7, 8, 2, 6, 1]
data2 = [3, 6, 8, 9, 3, 2, 10, 5, 6, 4, 8, 9, 3, 7, 2]

# Calculate quantiles
q1 = np.quantile(data1, np.linspace(0, 1, 100))
q2 = np.quantile(data2, np.linspace(0, 1, 100))

# Create the Q-Q plot
plt.plot(q1, q2, 'o', markersize=5)

# Reference line for perfect match (optional)
plt.plot(q1, q1, color='red', linestyle='--')

# Customize the plot (optional)
plt.xlabel('Quantiles of Data Set 1')
plt.ylabel('Quantiles of Data Set 2')
plt.title('Q-Q Plot of Sample Data Sets')
plt.grid(True)

# Display the plot
plt.show()



# Sample data (replace with your actual data)
data1 = [2, 5, 7, 10, 12]
data2 = [3, 6, 8, 9, 11, 13]

# Perform Wilcoxon Rank Sum Test
statistic, pvalue = stats.ranksums(data1, data2)

# Print test results
print("Test Statistic:", statistic)
print("p-value:", pvalue)

# Decide on rejecting the null hypothesis based on significance level (e.g., 0.05)
if pvalue < 0.05:
    print("Reject null hypothesis: There is a significant difference between the distributions.")
else:
    print("Fail to reject null hypothesis: Insufficient evidence to conclude a difference.")



# Sample data (replace with your actual data)
data1 = [2, 5, 7, 10, 12]
data2 = [3, 6, 8, 9, 11, 13]
data3 = [1, 4, 6, 9, 10]

# Perform Kruskal-Wallis test
statistic, pvalue = stats.kruskal(*[data1, data2, data3])

# Print test results
print("Test Statistic:", statistic)
print("p-value:", pvalue)

# Decide on rejecting the null hypothesis based on significance level (e.g., 0.05)
if pvalue < 0.05:
    print("Reject null hypothesis: There is a significant difference between distributions.")
else:
    print("Fail to reject null hypothesis: Insufficient evidence to conclude a difference.")



# Sample contingency table (replace with your actual data)
observed_data = [[10, 20],
                 [15, 25]]

# Perform Chi-square test
chi2_statistic, pvalue, expected_counts, variance = stats.chi2_contingency(observed_data)

# Print test results
print("Chi-square statistic:", chi2_statistic)
print("p-value:", pvalue)

# Decide on rejecting the null hypothesis based on significance level (e.g., 0.05)
if pvalue < 0.05:
    print("Reject null hypothesis: There is a significant association between the variables.")
else:
    print("Fail to reject null hypothesis: Insufficient evidence to conclude an association.")



"""
Non-parametric statistics focus on analyzing data without making strong assumptions about the underlying
distribution. Python offers various methods for exploring data distributions, such as histograms,
kernel density estimation (KDE), and Q-Q plots. Apart from this, non-parametric hypothesis testing
techniques like the Wilcoxon rank-sum test, Kruskal-Wallis test, and chi-square test allow for inferential
analysis without relying on parametric assumptions.
"""
