import os
import numpy as np
import matplotlib
from matplotlib import pyplot
from scipy import optimize
import utils

def plotData(X, y):
    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], '*')
    pyplot.plot(X[neg, 0], X[neg, 1], 'o')
    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(['Admitted', 'Not admitted'])


def sigmoid(z):

    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta, X, y):
    m = y.size  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)

    h = sigmoid(np.dot(X, theta))
    J = np.sum(-np.dot(y, np.log(h)) - (1-y).dot(np.log(1-h)))/m
    grad = ((h - y).dot(X))/m

    return J, grad


def predict(theta, X):
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    p = np.dot(X, theta)
    p = (p>0.5).astype(int)

    return p


data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_2/ex2data1.txt', delimiter=',')
X, y = data[:, 0:2], data[0:, 2]

z = [0, 1]
g = sigmoid(z)

print('g(', z, ') = ', g)

m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)

initial_theta = np.zeros(n+1)
cost, grad = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))

options = {'maxiter': 400}
res = optimize.minimize(costFunction, initial_theta, (X, y), jac=True, method='TNC', options=options)
cost = res.fun
theta = res.x
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
# utils.plotDecisionBoundary(plotData, theta, X, y)

prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))

p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')