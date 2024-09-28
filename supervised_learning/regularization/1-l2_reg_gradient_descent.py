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
    
    # Compute the gradient for the output layer
    dZ = cache['A{}'.format(L)] - Y
    dW = (1 / m) * np.dot(dZ, cache['A{}'.format(L-1)].T) + (lambtha / m) * weights['W{}'.format(L)]
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    
    # Update weights and biases for the output layer
    weights['W{}'.format(L)] -= alpha * dW
    weights['b{}'.format(L)] -= alpha * db
    
    # Compute gradients and update weights for hidden layers
    for l in range(L-1, 0, -1):
        dA = np.dot(weights['W{}'.format(l+1)].T, dZ)
        dZ = np.multiply(dA, 1 - np.power(cache['A{}'.format(l)], 2))
        dW = (1 / m) * np.dot(dZ, cache['A{}'.format(l-1)].T) + (lambtha / m) * weights['W{}'.format(l)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Update weights and biases
        weights['W{}'.format(l)] -= alpha * dW
        weights['b{}'.format(l)] -= alpha * db
