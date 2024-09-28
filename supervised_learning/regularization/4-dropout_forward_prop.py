#!/usr/bin/env python3
"""dropout"""

import numpy as np


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
    outputs = {}
    outputs['A0'] = X
    for layer in range(L):
        # define the weights and biases
        weight = weights['W{}'.format(layer+1)]
        bias = weights['b{}'.format(layer+1)]
        # calculate layer output as before
        linear_reg = np.matmul(weight, outputs['A{}'.format(layer)])
        z = np.add(linear_reg, bias)
        # Get randomized 1 and 0's
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        #  check if layer is final
        if layer != L-1:
            # apply tanh activation
            A = np.tanh(z)
            # multiply the output of the layer to the dropout
            A *= dropout
            # scale A
            A /= keep_prob
            outputs['D{}'.format(layer+1)] = dropout
        else:
            # apply softmax activation for the output layer
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs['A{}'.format(layer+1)] = A
    return outputs
