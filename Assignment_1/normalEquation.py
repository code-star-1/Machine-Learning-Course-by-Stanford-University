import numpy as np


def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta


data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_1/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = normalEqn(X, y)
print('Theta computed from the normal equations: {:s}'.format(str(theta)));

price = 0
X = np.array([1, 1650, 3], float)
price = X.dot(theta)
print(price)