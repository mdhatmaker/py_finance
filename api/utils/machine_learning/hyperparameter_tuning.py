from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# https://www.askpython.com/python/examples/https-www-askpython-com-python-examples-random-search-in-machine-learning


iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



param_dist = {
    'n_estimators': randint(10, 1000),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)

best_model = random_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy of best model: ", accuracy)



"""
Random Search Technique Overview

Define the Hyperparameter Space: First, specify the hyperparameter search space. This includes the
parameters to tune and their possible values or ranges, based on the specific model used.

Randomly Sample Hyperparameters: Generate random combinations of hyperparameters by sampling from the
defined search space. Each combination represents a unique configuration for the model.

Train and Evaluate Models: For each sampled set of hyperparameters, train the model using the training
data and evaluate its performance on a validation set. The evaluation metric, such as accuracy or loss,
measures how well the model performs with the given hyperparameters.

Select the Best Model: After training and evaluating multiple models with different hyperparameter
combinations, choose the model that achieves the best performance based on the evaluation metric.
This model and its corresponding hyperparameters are considered the optimal configuration.
"""




