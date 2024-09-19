from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/one-class-svm-anomaly-detection
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

# class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)



# normal dataset
normaldata = np.random.normal(0, 1, (100, 2))
# Training the model on normal data
clf = svm.OneClassSVM(nu=0.1, kernel="rbf")
clf.fit(normaldata)
#testing data with both normal and anamolous datapoints
testnormal = np.random.normal(0, 1, (50, 2))
testanomalous = np.random.uniform(-5, 5, (10, 2))

#Visualization
plt.figure(figsize=(10, 6))
plt.scatter(normaldata[:, 0], normaldata[:, 1], label='Normal Data', color='green')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='r', label='Support Vectors')
plt.scatter(testnormal[:, 0], testnormal[:, 1], label='Test Data - Normal', color='blue')
plt.scatter(testanomalous[:, 0], testanomalous[:, 1], label='Test Data - Anomalous', color='red')
plt.title('1-Class SVM for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


