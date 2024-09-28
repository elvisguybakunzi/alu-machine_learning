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
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (1 - np.square(A))
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
