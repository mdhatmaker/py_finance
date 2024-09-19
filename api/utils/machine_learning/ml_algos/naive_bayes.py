from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# https://www.askpython.com/python/examples/top-machine-learning-algorithms
# https://www.askpython.com/python/examples/naive-bayes-classifier


# Loading the Dataset
data_loaded = load_breast_cancer()
X = data_loaded.data
y = data_loaded.target

# Splitting the dataset into training and testing variables
X_train, X_test, y_train, y_test = train_test_split(data_loaded.data, data_loaded.target, test_size=0.2, random_state=20)
# keeping 80% as training data and 20% as testing data.

# Importing the Gaussian Naive Bayes Class and fitting the training data to it.
# Calling the Class
naive_bayes = GaussianNB()
# Fitting the data to the classifier
naive_bayes.fit(X_train, y_train)
# Predict on test data
y_predicted = naive_bayes.predict(X_test)

# Now, let’s find how accurate our model was using accuracy metrics.
metrics.accuracy_score(y_predicted, y_test)





"""
Bayes theorem gives us the probability of Event A to happen given that event B has occurred. For example.

What is the probability that it will rain given that its cloudy weather? The probability of rain can be called as our hypothesis and the event representing cloudy weather can be called as evidence.

P(A|B) – is called as a posterior probability
P(B|A) – is the conditional probability of B given A.
P(A) – is called as Prior probability of event A.
P(B) – regardless of the hypothesis, it is the probability of event B to occur.

Types of Naïve Bayes Classifier:

Multinomial – It is used for Discrete Counts. The one we described in the example above is an example of Multinomial Type Naïve Bayes.
Gaussian – This type of Naïve Bayes classifier assumes the data to follow a Normal Distribution.
Bernoulli – This type of Classifier is useful when our feature vectors are Binary.

"""