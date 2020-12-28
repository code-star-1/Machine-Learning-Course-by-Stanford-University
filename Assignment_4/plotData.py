import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    # Setup some useful variables
    m = y.size

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    #dimensions
    """
    a1 = 5000 * 401
    theta1 = 25 * 401
    theta2 = 10 * 26
    a2 = 5000 * 26
    a3 = 5000 * 10
    y_matrix = 5000 * 10
    delta_3 = 5000 * 10
    delta_2 = 5000 * 25
    Delta1 = 25 * 401
    Delta2 = 10 * 26
    """
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    a2 = utils.sigmoid(np.dot(a1, Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    a3 = utils.sigmoid(np.dot(a2, Theta2.T))
    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]
    temp1 = Theta1
    temp2 = Theta2
    regularized_term = (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))*(lambda_/(2*m))
    J = (-1/m)*np.sum((np.log(a3)*y_matrix) + (np.log(1-a3))*(1-y_matrix)) + regularized_term

    delta_3 = a3 - y_matrix
    delta_2 = delta_3.dot(Theta2)[:, 1:] * utils.sigmoidGradient(a1.dot(Theta1.T))
    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)
    Theta1_grad = (1/m)*Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_/m)*Theta1_grad[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2_grad[:, 1:]
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

data = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_4/ex4data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y==10] = 0
m = y.size

rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
# utils.displayData(sel)

input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

weights = loadmat(os.path.join('Data', '/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_4/ex4weights.mat'))
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
lambda_ = 0
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
lambda_ = 1
J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

print('Cost at parameters (loaded from ex4weights): %.6f' % J)

lambda_ = 3
utils.checkNNGradients(nnCostFunction, lambda_)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, lambda_)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))

options = {'maxiter': 100}

#  You should also try different values of lambda
lambda_ = 3

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))


pred = utils.predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

utils.displayData(Theta1[:, 1:])