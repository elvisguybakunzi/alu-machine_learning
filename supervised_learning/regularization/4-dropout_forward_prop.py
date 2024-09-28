#!/usr/bin/env python3
"""dropout"""

import numpy as np


def softmax(Z):
    """Softmax activation function."""
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / exp_Z.sum(axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Arguments:
    - X: numpy.ndarray of shape (nx, m) containing the input data
    - weights: dictionary of the weights and biases of the neural network
    - L: the number of layers in the network
    - keep_prob: the probability that a node will be kept

    Returns:
    - A dictionary containing the outputs of
    each layer and the dropout mask used on each layer.
    """
    cache = {}
    cache['A0'] = X

    for le in range(1, L + 1):
        Wl = weights['W' + str(le)]
        bl = weights['b' + str(le)]
        A_prev = cache['A' + str(le - 1)]

        # Linear step
        Zl = np.matmul(Wl, A_prev) + bl

        if le == L:
            # Last layer uses softmax activation
            Al = softmax(Zl)
        else:
            # Hidden layers use tanh activation
            Al = np.tanh(Zl)

            # Dropout mask for layer l
            Dl = np.random.rand(Al.shape[0], Al.shape[1]) < keep_prob
            Al *= Dl  # Apply dropout
            Al /= keep_prob  # Inverted dropout

            cache['D' + str(le)] = Dl  # Save dropout mask for layer l

        cache['A' + str(le)] = Al  # Save activation output for layer l

    return cache
