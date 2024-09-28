#!/usr/bin/env python3
"""update the weights"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network
    with Dropout regularization using gradient descent"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            D = cache['D' + str(i - 1)]  # Dropout mask
            dA_prev = np.dot(W.T, dZ)
            dA_prev *= D  # Apply dropout mask
            dA_prev /= keep_prob  # Scale dropout
            dZ = dA_prev * (1 - np.power(A_prev, 2))  # Derivative of tanh

        # Update weights and biases
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
