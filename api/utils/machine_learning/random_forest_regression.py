import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# https://www.askpython.com/python/examples/random-forest-regression


def plot_result(X, y, regressor):
    # higher resolution graph
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(X, y, color='red')  # plotting real points
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # plotting for predict points
    plt.title("Truth or Bluff(Random Forest - Smooth)")
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()


def run_random_forest_regression(dataset):
    """
    2. Data preprocessing

    We will not have much data preprocessing. We will just have to identify the matrix of features
    and the vectorized array.
    """
    X = dataset.iloc[:,1:2].values
    y = dataset.iloc[:,2].values


    """
    3. Fitting the Random forest regression to dataset
    
    We will import the RandomForestRegressor from the ensemble library of sklearn. We create a regressor
    object using the RFR class constructor.
    
    The parameters include:
    1. n_estimators : number of trees in the forest. (default = 10)
    2. criterion : Default is mse ie mean squared error. This was also a part of decision tree.
    3. random_state
    """
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X,y)
    plot_result(X, y, regressor)      # Visualize the result

    # Make a test prediction
    y_pred=regressor.predict([[6.5]])
    print(y_pred)


    # Rebuild the model for 100 trees
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X,y)
    plot_result(X, y, regressor)      # Visualize the result

    # Make another test prediction
    y_pred=regressor.predict([[6.5]])
    print(y_pred)


    # Rebuild the model for 300 trees
    regressor = RandomForestRegressor(n_estimators=300, random_state=0)
    regressor.fit(X,y)
    plot_result(X, y, regressor)      # Visualize the result

    # Make another test prediction
    y_pred=regressor.predict([[6.5]])
    print(y_pred)


if __name__ == "__main__":
    file_dir = "/files/in"
    dataset = pd.read_csv(f'{file_dir}/Position_Salaries.csv')
    dataset.head()
    run_random_forest_regression(dataset)


"""
What is Regression in Machine Learning?

Regression is a machine learning technique that is used to predict values across a certain range. Let us see understand this concept with an example, consider the salaries of employees and their experience in years.

A regression model on this data can help in predicting the salary of an employee even if that year is not having a corresponding salary in the dataset.

What is Random Forest Regression?

Random forest regression is an ensemble learning technique. But what is ensemble learning?

In ensemble learning, you take multiple algorithms or same algorithm multiple times and put together a model that’s more powerful than the original.

Prediction based on the trees is more accurate because it takes into account many predictions. This is because of the average value used. These algorithms are more stable because any changes in dataset can impact one tree but not the forest of trees.

Steps to perform the random forest regression

This is a four step process and our steps are as follows:

1. Pick a random K data points from the training set.
2. Build the decision tree associated to these K data points.
3. Choose the number N tree of trees you want to build and repeat steps 1 and 2.
4. For a new data point, make each one of your Ntree trees predict the value of Y for the data point in the question, and assign the new data point the average across all of the predicted Y values.

Implementing Random Forest Regression in Python

Our goal here is to build a team of decision trees, each making a prediction about the dependent variable and the ultimate prediction of random forest is average of predictions of all trees.

For our example, we will be using the Salary – positions dataset which will predict the salary based on prediction.

The dataset used can be found at https://github.com/content-anu/dataset-polynomial-regression
"""
