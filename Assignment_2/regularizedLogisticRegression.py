import numpy as np
from matplotlib import pyplot
import utils
from scipy import optimize

def sigmoid(z):

    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

def plotData(X, y):
    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], '*')
    pyplot.plot(X[neg, 0], X[neg, 1], 'o')
    pyplot.xlabel('Microchip Test 1')
    pyplot.ylabel('Microchip Test 2')
    pyplot.legend(['y = 1', 'y = 0'], loc='upper right')


def costFunctionReg(theta, X, y, lambda_):
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    H = sigmoid(np.dot(X, theta))
    temp = theta
    temp[0] = 0
    J = np.sum(-np.dot(y, np.log(H)) - np.dot(1-y, np.log(1-(H))))/m + (lambda_*np.sum(np.square(temp)))/(2*m)
    grad = ((H - y).dot(X))/m + (lambda_/m)*temp

    return J, grad

def predict(theta, X):
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    p = np.dot(X, theta)
    p = (p>0.5).astype(int)

    return p


data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_2/ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
plotData(X, y)

X = utils.mapFeature(X[:, 0], X[:, 1])

initial_theta = np.zeros(X.shape[1])
lambda_ = 1
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)
print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))

test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)
print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Gradient at test theta - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))

initial_theta = np.zeros(X.shape[1])
lambda_ = 1
options= {'maxiter': 100}
res = optimize.minimize(costFunctionReg, initial_theta, (X, y, lambda_), jac=True, method='TNC', options=options)
cost = res.fun
theta = res.x

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')

utils.plotDecisionBoundary(plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)