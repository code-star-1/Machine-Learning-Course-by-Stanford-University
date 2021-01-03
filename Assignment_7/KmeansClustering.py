import os
import numpy as np
import re
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from IPython.display import HTML, display, clear_output

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

from scipy import optimize
from scipy.io import loadmat
import utils


def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    for i in np.arange(idx.size):
        J = np.sqrt(np.sum(np.square(X[i] - centroids), axis=1))
        idx[i] = np.argmin(J)
    return idx


def computeCentroids(X, idx, K):
    # Useful variables
    m, n = X.shape
    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))
    for i in np.arange(K):
        centroids[i] = np.mean(X[idx == i], axis=0)
    return centroids


def kMeansInitCentroids(X, K):
    m, n = X.shape

    # You should return this values correctly
    centroids = np.zeros((K, n))
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K], :]
    return centroids




data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_7/ex7data2.mat'))
X = data['X']
#
# K = 3
# max_iters = 10


# initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])


# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
# centroids, idx, anim = utils.runkMeans(X, initial_centroids,
#                                        findClosestCentroids, computeCentroids, max_iters, True)

# ======= Experiment with these parameters ================
# You should try different values for those parameters
K = 100
max_iters = 10

# Load an image of a bird
# Change the file name and path to experiment with your own images
A = mpl.image.imread(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_7/bird_small.png'))
# ==========================================================

# Divide by 255 so that all values are in the range 0 - 1
A /= 255

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(-1, 3)

# When using K-Means, it is important to randomly initialize centroids
# You should complete the code in kMeansInitCentroids above before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = utils.runkMeans(X, initial_centroids,
                                 findClosestCentroids,
                                 computeCentroids,
                                 max_iters)

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
# Reshape the recovered image into proper dimensions
X_recovered = centroids[idx, :].reshape(A.shape)

# Display the original image, rescale back by 255
fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(A*255)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed, with %d colors' % K)
ax[1].grid(False)
pyplot.show()

