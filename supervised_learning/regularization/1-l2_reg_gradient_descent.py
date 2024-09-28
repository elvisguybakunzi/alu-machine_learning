#!/usr/bin/env python3
"""Gradient descent"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot encoded matrix of shape (classes, m)
    containing the correct labels for the data.
    classes: number of classes, m: number of data points.
    weights (dict): Dictionary of the weights and
    biases of the neural network.
    cache (dict): Dictionary of the outputs of each layer
    of the neural network.
    alpha (float): Learning rate.
    lambtha (float): L2 regularization parameter.
    L (int): Number of layers of the network.

    The weights and biases are updated in place.
    """
    m = Y.shape[1]  # Number of data points
    A_last = cache['A' + str(L)]  # Output of the last layer

    # Derivative of cost with respect to Z for the output layer
    dZ = A_last - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]  # Output of the previous layer
        W = weights['W' + str(l)]  # Weights of the current layer
        b = weights['b' + str(l)]  # Biases of the current layer

        # Gradient of weights and biases with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            # For layers before the last one, calculate dZ for the iteration
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh activation
