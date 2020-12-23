import numpy as np
import matplotlib.pyplot as pyplot

def printDataPoint(X, y):
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    print('-' * 26)
    for i in range(10):
        print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0
    H= X.dot(theta)
    J = np.sum((H-y)**2)/(2*m)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()

    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha/m) * (X.dot(theta)-y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_1/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

alpha = 0.3
num_iters = 400
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
# pyplot.show()
print('theta computed from gradient descent: {:s}'.format(str(theta)))
X_array = np.array([1, 1650, 3], dtype=float)
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))
