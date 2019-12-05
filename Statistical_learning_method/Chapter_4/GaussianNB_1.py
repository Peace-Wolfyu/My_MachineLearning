# -*- coding: utf-8 -*-

# @Time  : 2019/12/5 14:14

# @Author : Mr.Lin

from collections import Counter

import numpy as np
from numpy import ndarray, exp, pi, sqrt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain


class GaussianNB:
    """GaussianNB class support multiple classification.
    Attributes:
        prior: Prior probability.
        avgs: Means of training data. e.g. [[0.5, 0.6], [0.2, 0.1]]
        vars: Variances of training data.
        n_class: number of classes
    """

    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    @staticmethod
    def _get_prior(label: ndarray) -> ndarray:
        """Calculate prior probability.
        Arguments:
            label {ndarray} -- Target values.
        Returns:
            array
        """

        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    def _get_avgs(self, data: ndarray, label: ndarray) -> ndarray:
        """Calculate means of training data.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        Returns:
            array
        """
        return np.array([data[label == i].mean(axis=0)
                         for i in range(self.n_class)])

    def _get_vars(self, data: ndarray, label: ndarray) -> ndarray:
        """Calculate variances of training data.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        Returns:
            array
        """
        return np.array([data[label == i].var(axis=0)
                         for i in range(self.n_class)])


    # 高斯分布

    def _get_posterior(self, row: ndarray) -> ndarray:
        """Calculate posterior probability
        Arguments:
            row {ndarray} -- Sample of training data.
        Returns:
            array
        """

        return (1 / sqrt(2 * pi * self.vars) * exp(
            -(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)

    def fit(self, data: ndarray, label: ndarray):
        """Build a Gauss naive bayes classifier.
        Arguments:
            data {ndarray} -- Training data.
            label {ndarray} -- Target values.
        """

        # Calculate prior probability.
        self.prior = self._get_prior(label)
        # Count number of classes.
        self.n_class = len(self.prior)
        # Calculate the mean.
        self.avgs = self._get_avgs(data, label)
        # Calculate the variance.
        self.vars = self._get_vars(data, label)

    def predict_prob(self, data: ndarray) -> ndarray:
        """Get the probability of label.
        Arguments:
            data {ndarray} -- Testing data.
        Returns:
            ndarray -- Probabilities of label.
            e.g. [[0.02, 0.03, 0.02], [0.02, 0.03, 0.02]]
        """

        # Caculate the joint probabilities of each feature and each class.
        likelihood = np.apply_along_axis(self._get_posterior, axis=1, arr=data)
        probs = self.prior * likelihood
        # Scale the probabilities
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.
        Arguments:
            data {ndarray} -- Testing data.
        Returns:
            ndarray -- Prediction of label.
        """

        # Choose the class which has the maximum probability
        return self.predict_prob(data).argmax(axis=1)

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

def _clf_input_check(y, y_hat):
    m = len(y)
    n = len(y_hat)
    elements = chain(y, y_hat)
    valid_elements = {0, 1}
    assert m == n, "Lengths of two arrays do not match!"
    assert m != 0, "Empty array!"
    assert all(element in valid_elements
               for element in elements), "Array values have to be 0 or 1!"
def _get_acc(y, y_hat):
    """Calculate the prediction accuracy.
    Arguments:
        y {ndarray} -- 1d array object with int.
        y_hat {ndarray} -- 1d array object with int.
    Returns:
        float
    """

    _clf_input_check(y, y_hat)
    return (y == y_hat).sum() / len(y)



X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = GaussianNB()
clf.fit(X_train, y_train)
# Model evaluation
y_hat = clf.predict(X_test)
print(y_hat)
acc = _get_acc(y_test, y_hat)
print("Accuracy is %.3f" % acc)
