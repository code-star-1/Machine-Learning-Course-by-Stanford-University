import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):

    m = y.size  # number of training examples
    J = 0
    H = np.dot(X, theta)
    J = np.sum((H-y)**2)/(2*m)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):

    m = y.shape[0]  # number of training examples
    theta = theta.copy()

    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

def visualize(X, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = computeCost(X, y, [theta0, theta1])


    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # surface plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Surface')

    # # contour plot
    # # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax = plt.subplot(122)
    plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.plot(theta0_vals[0], theta0_vals[1], 'ro', ms=10, lw=2)
    plt.title('Contour, showing minimum')
    plt.show()
    pass


def main():
    data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_1/ex1data1.txt', delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m = len(y)
    # axis = 0 means columnwise, axis = 1 means rowwise https://stackoverflow.com/questions/22149584/what-does-axis-in-pandas-mean
    X = np.stack([np.ones(m), X], axis=1)

    J = computeCost(X, y, theta=np.array([0.0, 0.0]))
    print('With theta = [0, 0] \nCost computed = %.2f' % J)

    J = computeCost(X, y, theta=np.array([-1, 2]))
    print('With theta = [-1, 2]\nCost computed = %.2f' % J)

    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
    # plt.plot(X[:, 1], y, 'o')
    # plt.plot(X[:, 1], np.dot(X, theta), '-')
    # plt.legend(['Training data', 'Linear regression']);

    predict1 = np.dot([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of {:.2f}'.format(predict1 * 10000))

    predict2 = np.dot([1, 7], theta)
    print('For population = 70,000, we predict a profit of {:.2f}'.format(predict2 * 10000))

    # visualize(X, y)

main()
