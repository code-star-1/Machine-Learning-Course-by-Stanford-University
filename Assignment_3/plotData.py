import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils


def lrCostFunction(theta, X, y, lambda_):
    m = y.size

    if y.dtype == bool:
        y = y.astype(int)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    H = np.dot(X, theta)
    H = utils.sigmoid(H)
    theta[0] = 0
    J = (np.dot(-y, np.log(H)) - np.dot(1-y, np.log(1-H)))/m + lambda_*np.sum(theta**2)/(2*m)
    grad = np.dot(H-y, X)/m + lambda_*(theta)/m
    return J, grad

def oneVsAll(X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n+1, dtype=float)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, (y == c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)
        cost = res.fun
        theta = res.x
        all_theta[c, :] = theta

    return all_theta


def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    p = np.argmax(utils.sigmoid(np.dot(X, all_theta.T)), axis=1)
    return p

input_layer_size  = 400
num_labels = 10
data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_3/ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

y[y == 10] = 0
m = y.size

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

# utils.displayData(sel)
# pyplot.show()

theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print('Cost         : {:.6f}'.format(J))
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

pred = predictOneVsAll(all_theta, X)

print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))
