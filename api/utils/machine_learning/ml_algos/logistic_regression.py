import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# https://www.askpython.com/python/examples/top-machine-learning-algorithms
# https://www.askpython.com/python/examples/logistic-regression-from-scratch
# https://www.askpython.com/python/examples/logistic-regression
# https://github.com/Ash007-kali/Article-Datasets/tree/main/Logistic%20Regression%20From%20Scratch


data = pd.read_csv("bank-loan.csv")     # dataset

X = loan.drop(['default'], axis=1)
Y = loan['default'].astype(str)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=0)


def err_metric(CM):
    TN = CM.iloc[0, 0]
    FN = CM.iloc[1, 0]
    TP = CM.iloc[1, 1]
    FP = CM.iloc[0, 1]
    precision = (TP) / (TP + FP)
    accuracy_model = (TP + TN) / (TP + TN + FP + FN)
    recall_score = (TP) / (TP + FN)
    specificity_value = (TN) / (TN + FP)

    False_positive_rate = (FP) / (FP + TN)
    False_negative_rate = (FN) / (FN + TP)

    f1_score = 2 * ((precision * recall_score) / (precision + recall_score))

    print("Precision value of the model: ", precision)
    print("Accuracy of the model: ", accuracy_model)
    print("Recall value of the model: ", recall_score)
    print("Specificity of the model: ", specificity_value)
    print("False Positive rate of the model: ", False_positive_rate)
    print("False Negative rate of the model: ", False_negative_rate)
    print("f1 score of the model: ", f1_score)



logit = LogisticRegression(class_weight='balanced' , random_state=0).fit(X_train,Y_train)
target = logit.predict(X_test)
CM_logit = pd.crosstab(Y_test,target)
err_metric(CM_logit)


# Defining a sigmoid function
def sigmoid(z):
    op = 1/(1 + np.exp(-z))
    return op


# Loss Function
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# Gradient_descent
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


class LogisticRegression:
    def __init__(self, x, y):
        self.intercept = np.ones((x.shape[0], 1))
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.zeros(self.x.shape[1])
        self.y = y

    # Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))

    # method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, lr, iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)

            loss = self.loss(sigma, self.y)

            dW = self.gradient_descent(self.x, sigma, self.y)

            # Updating the weights
            self.weight -= lr * dW

        return print('fitted successfully to data')

    # Method to predict the class label.
    def predict(self, x_new, treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True:
                y_pred[i] = 1
            else:
                continue

        return y_pred


def test_implementation():
    # Loading the data
    data = load_breast_cancer()

    # Preparing the data
    x = data.data
    y = data.target

    # creating the class Object
    regressor = LogisticRegression(x, y)

    #
    regressor.fit(0.1, 5000)

    y_pred = regressor.predict(x, 0.5)

    print('accuracy -> {}'.format(sum(y_pred == y) / y.shape[0]))


