#!/usr/bin/env python3
"""L2 regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (float): Cost of the network without L2 regularization.
    lambtha (float): Regularization parameter.
    weights (dict): Dictionary of the weights and biases
    (numpy.ndarrays) of the neural network.
    L (int): Number of layers in the neural network.
    m (int): Number of data points used.

    Returns:
    float: Cost of the network accounting for L2 regularization.
    """
    l2_reg_term = 0

    # Sum the Frobenius norm (squared L2 norm) of the weights for each layer
    for i in range(1, L + 1):
        l2_reg_term += np.sum(np.square(weights['W' + str(i)]))

    # Add the L2 regularization term to the original cost
    l2_reg_cost = cost + (lambtha / (2 * m)) * l2_reg_term

    return l2_reg_cost
