from scipy.stats import chi2_contingency
import os
import pandas as pd
import numpy as np

# https://www.askpython.com/python/examples/chi-square-test


info = [[100, 200, 300], [50, 60, 70]]
print(info)
stat, p, dof = chi2_contingency(info)

print(dof)

significance_level = 0.05
print("p value: " + str(p))
if p <= significance_level:
    print('Reject NULL HYPOTHESIS')
else:
    print('ACCEPT NULL HYPOTHESIS')



#Changing the current working directory
os.chdir("D:/Ediwsor_Project - Bike_Rental_Count")
BIKE = pd.read_csv("day.csv")
categorical_col = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
print(categorical_col)
chisqt = pd.crosstab(BIKE.holiday, BIKE.weathersit, margins=True)
print(chisqt)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])







