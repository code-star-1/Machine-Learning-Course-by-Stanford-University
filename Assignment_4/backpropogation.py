import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

def sigmoidGradient(z):

    g = np.zeros(z.shape)
    g = utils.sigmoid(z)*(1-utils.sigmoid(z))
    return g


def randInitializeWeights(L_in, L_out, epsilon_init=0.12):

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W

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
    Theta2_grad = (1/m)*Delta2
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_/m)*Theta1_grad[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2_grad[:, 1:]
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad


z = np.array([-1, -0.5, 0, 0.5, 1])
g = sigmoidGradient(z)

print('Initializing Neural Network Parameters ...')

input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

utils.checkNNGradients(nnCostFunction)

lambda_ = 3
utils.checkNNGradients(nnCostFunction, lambda_)

