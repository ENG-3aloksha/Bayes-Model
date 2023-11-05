import functions
from sklearn import datasets
import numpy as np


"""
This module to train and test bayes model classifier on iris dataset
"""

iris = datasets.load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

x_train, x_validate, x_test, y_train, y_validate, y_test = functions.train_validate_test_split(X, Y, 0.2, 0.4)

# print("x_train : {}".format(x_train.shape))
# print("x_validate : {}".format(x_validate.shape))
# print("x_test : {}".format(x_test.shape))
# print("y_train : {}".format(y_train.shape))
# print("y_validate : {}".format(y_validate.shape))
# print("y_test : {}".format(y_test.shape))
print(type(x_train))