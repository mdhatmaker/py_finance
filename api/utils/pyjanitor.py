import pandas as pd
import numpy as np
import janitor

# https://www.askpython.com/python-modules/pyjanitor-miscellaneous-functions
# https://pyjanitor-devs.github.io/pyjanitor/api/functions/#janitor.functions/


# General Functions in PyJanitor

# 1. Count Cumulative Unique
# Remember that in lower-grade mathematics, we used to compute the cumulative frequency?
# The same concept applies here. The count cumulative unique function returns a column containing
# the cumulative sum of unique values in the specified column.

# count_cumulative_unique(df, column_name, dest_column_name, case_sensitive=True)

df = pd.DataFrame({
    "letters": list("abABcdef"),
    "numbers": range(4, 12),
})
print(df)

df.count_cumulative_unique(
    column_name="letters",
    dest_column_name="letters_count",
    case_sensitive = True,
)

df.count_cumulative_unique(
    column_name="letters",
    dest_column_name="letters_count",
    case_sensitive = False,
)


# 2. Drop Constant Columns
# This function is used to drop or remove all the columns that have constant(same) values.

# drop_constant_columns(df)

data = {'A':[3,3,3,],
        'B':[3,2,1],
        'C':[3,1,2],
        'D':["Noodles","China","Japan"],
        'E':["Pao","China","Kimchi"],
        'F':["Japan","China","Korea"]}
df = pd.DataFrame(data)
print(df)

df.drop_constant_columns()

# 3. Drop Duplicate Columns
# This method is useful when there are multiple columns with the same name.
# In such cases, we can specify the column name and the index of the column such that the repetitive
# column at that index will be dropped.

# drop_duplicate_columns(df, column_name, nth_index=0)

df = pd.DataFrame({
    "a": range(2, 5),
    "b": range(3, 6),
    "A": range(4, 7),
    "b*": range(6, 9),
}).clean_names(remove_special=True)
print(df)

df.drop_duplicate_columns(column_name="b", nth_index=0)


# 4.Find_Replace
# The find_replace function just as its name suggests, is used to find an element in the dataframe
# and replace it with some other element.

# find_replace(df, match='exact', **mappings)

df = pd.DataFrame({
    "song": ["We don't talk anymore","Euphoria","Dangerously","As It Was"],
    "singer": ["C.Puth","JK","C.Puth","Harry Styles"]
})
print(df)

df = find_replace(
    df,
    match="exact",
    singer={"C.Puth":"Charile","JK":"Jungkook"},
)
print(df)


# 5. Jitter
# Jitter is a function of PyJanitor that can be used to introduce noise to the values of the
# data frame. If the data frame has NaN values, they are ignored and the jitter value corresponding
# to this element will also be NaN.

# jitter(df, column_name, dest_column_name, scale, clip=None, random_state=None)

df1 = pd.DataFrame({"a": [3, 4, 5, np.nan], "b":[1,2,3,4]})
print(df1)

df1.jitter("a", dest_column_name="jit", scale=2,random_state=0)



# Math Functions in PyJanitor

# 1. Ecdf
# The ecdf is a function used to obtain the empirical cumulative distribution of values in a series.
# Given a series as an input, this function generates a sorted array of values in the series and
# computes a cumulative fraction of data points with values less or equal to the array.

# ecdf(s)

s = pd.Series([5,1,3,4,2])
x,y= janitor.ecdf(s)
print("The sorted array of values:",x)
print("The values less than equal to x:",y)

# 2. Exponent
# The exp(s) takes a series as input and returns the exponential for each value in the series.

# exp(s)

s = pd.Series([1,2,7,6,5])
exp_values = s.exp()
print(exp_values)


# 3. Sigmoid
# The sigmoid function of pyjanitor is used to compute the sigmoid values for each element in the series.

# sigmoid(x) = 1 / (1 + exp(-x))

s = pd.Series([1, 2, 4, -3])
sigvalues = s.sigmoid()
print(sigvalues)


# 4. Softmax
# The softmax function, just as the name suggests is used to compute the softmax values for the
# elements in a series or a one-dimensional numpy array.

# softmax(x) = exp(x)/sum(exp(x))

s = pd.Series([1,-2,5])
s.softmax()


# 5. Z-Score
# Z-score is an important parameter in statistics and even in the field of machine learning.
# Also called the standards score, it is used to describe the relationship of a value to the mean
# of the group of values.

# z = (s - s.mean()) / s.std()

s = pd.Series([0, 1, 3,9,-2])
s.z_score()













"""
PyJanitor, a data cleaning and processing API built on top of the Pandas library, offers a wide range
of miscellaneous functions for various domains such as finance, engineering, biology, and time series
analysis. These functions include general utilities like counting cumulative unique values, dropping
constant or duplicate columns, finding and replacing elements, and introducing noise with jitter.
Apart from them, PyJanitor provides math functions for computing empirical cumulative distribution,
exponentiation, sigmoid, softmax, and z-score standardization, making it a versatile tool for data
cleaning and manipulation tasks.
"""



