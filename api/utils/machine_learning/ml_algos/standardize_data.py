from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# https://www.askpython.com/python/examples/standardize-data-in-python


# By standardizing the values, we get the following statistics of the data distribution:
# - mean = 0
# - standard deviation = 1


# 1. Using preprocessing.scale() function
def preprocess_scale():
    data = load_iris()

    # separate the independent and dependent variables
    X_data = data.data
    target = data.target

    # standardization of dependent variables
    standard = preprocessing.scale(X_data)
    print(standard)


# 2. Using StandardScaler() function
def preprocess_standardscaler():
    data = load_iris()
    scale = StandardScaler()

    # separate the independent and dependent variables
    X_data = data.data
    target = data.target

    # standardization of dependent variables
    scaled_data = scale.fit_transform(X_data)
    print(scaled_data)

