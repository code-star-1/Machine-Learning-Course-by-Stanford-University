import numpy as np
from matplotlib import pyplot
import re
from scipy import optimize
from scipy.io import loadmat
import utils
import os


def gaussianKernel(x1, x2, sigma):
    sim = 0
    sim = np.exp(-np.sum((x1-x2)**2)/(2*sigma**2))
    return sim


def dataset3Params(X, y, Xval, yval):
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    C_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    error_array = np.zeros([len(C_array), len(sigma_array)], dtype=float)

    for i, c in enumerate(C_array):
        for j, sigma in enumerate(sigma_array):
            model = utils.svmTrain(X, y, C_array[i], gaussianKernel, args=(sigma_array[j],))
            predictions = utils.svmPredict(model, Xval)
            error_array[i, j] = np.mean(predictions != yval)
    ind = np.unravel_index(np.argmin(error_array), error_array.shape)
    C = C_array[ind[0]]
    sigma = sigma_array[ind[1]]
    return C, sigma


data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/ex6data1.mat'))
X, y = data['X'], data['y'][:, 0]

# Plot training data
# utils.plotData(X, y)
# pyplot.show()

C = 1
model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
# utils.visualizeBoundaryLinear(X, y, model)
# pyplot.show()

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
      '\t%f' % (sigma, sim))

data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/ex6data2.mat'))
X, y = data['X'], data['y'][:, 0]

# utils.plotData(X, y)
# pyplot.show()

C = 1
sigma = 0.1

# model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
# utils.visualizeBoundary(X, y, model)
# pyplot.show()
data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/ex6data3.mat'))
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# # Plot training data
# utils.plotData(X, y)
# pyplot.show()

C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
# model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
pyplot.show()
print(C, sigma)
