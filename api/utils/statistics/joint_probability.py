import numpy as np
import seaborn as sns
import pandas as pd

# https://www.askpython.com/python/examples/joint-probability-distribution


A = np.random.normal(size=100)
B = np.random.normal(size=100)

df = pd.DataFrame({'A' : A , 'B':B})

sns.jointplot(x='A', y='B' ,data=df )





