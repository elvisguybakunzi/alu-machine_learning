#!/usr/bin/env python3
"""Script that defines a deep neural network
with binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network
    with binary classification.
    """

    def __init__(self, nx, layers):
        """class constructor

        Args:
            nx (int): is the number of input features
            layers (list): is a list representing the number
            of nodes in each layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Check if layers is a list of positive integers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layers, int) and layers > 0
                   for layers in layers):
            raise ValueError("layers must be a list of positive integers")

        # Set public attributes

        # number of layers in the neural network
        self.L = len(layers)
        # Cache to hold intermediary values
        self.cache = {}
        # Dictionary to hold weight and biases
        self.weights = {}

        # Initialize weights and biases using He et al. method
        for le in range(1, self.L + 1):
            if le == 1:
                self.weights['W1'] = np.random.randn(
                  layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W{}'.format(le)] = np.random.randn(
                  layers[le - 1], layers[le - 2]) * np.sqrt(2 / layers[le - 2])
            self.weights['b{}'.format(le)] = np .zeros((layers[le - 1], 1))
