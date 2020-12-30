import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

def linearRegCostFunction(X, y, theta, lambda_=0.0):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    J = np.sum(np.square(np.dot(X, theta)-y))/(2*m) + (lambda_ / (2*m))*np.sum(np.square(theta[1:]))
    grad = (1/m) * (np.dot(X.T, np.dot(X, theta)-y))
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

    return J, grad

def learningCurve(X, y, Xval, yval, lambda_=0):

    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m+1):
        theta = utils.trainLinearReg(linearRegCostFunction, X[:i], y[:i], lambda_, maxiter=200)
        error_train[i-1], _ = linearRegCostFunction(X[:i, :], y[:i], theta, lambda_)
        error_val[i-1], _ = linearRegCostFunction(Xval, yval, theta, lambda_)

    return error_train, error_val


def polyFeatures(X, p):
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i] = (X[:, 0] ** (i+1))

    return X_poly


def validationCurve(X, y, Xval, yval):
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i, lambda_ in enumerate(lambda_vec):
        theta = utils.trainLinearReg(linearRegCostFunction, X, y, lambda_)
        error_train[i], _ = linearRegCostFunction(X, y, theta, lambda_ = 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, lambda_ = 0)

    return lambda_vec, error_train, error_val


data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_5/ex5data1.mat'))
X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

m = y.size

# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')

theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)
print('Cost at theta = [1, 1]:\t   %f ' % J)
print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))

X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = utils.trainLinearReg(linearRegCostFunction, X_aug, y, lambda_=0)

#  Plot fit over the data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.plot(X, np.dot(X_aug, theta), '--', lw=2)

X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

# pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
# pyplot.title('Learning curve for linear regression')
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 150])
# pyplot.show()

# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = utils.featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

print('Normalized Training Example 1:')
print(X_poly[0, :])

lambda_ = 100
theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
                             lambda_=lambda_, maxiter=55)

# Plot training data and fit
# pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

# utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)
#
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
# pyplot.ylim([-20, 50])
# pyplot.show()

# pyplot.figure()
# error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
# pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

# pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 100])
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.show()

# print('Polynomial Regression (lambda = %f)\n' % lambda_)
# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('lambda')
pyplot.ylabel('Error')
pyplot.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))