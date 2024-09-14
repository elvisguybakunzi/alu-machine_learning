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
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(
          map(lambda layer: isinstance(layer, int) and layer > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for le in range(1, self.L + 1):
            layer_size = layers[le - 1]
            input_size = nx if le == 1 else layers[le - 2]

            self.__weights['W' + str(le)] = np.random.randn(
              layer_size, input_size) * np.sqrt(2 / input_size)
            self.__weights['b' + str(le)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights"""
        return self.__weights

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Calculates the forward propagation
        of the neural network

        Args:
            X (array): is a numpy.ndarray
            with shape (nx, m) that contains
            the input data
        """

        # Input layer stored as A0
        self.__cache['A0'] = X

        for le in range(1, self.__L + 1):
            W = self.__weights['W' + str(le)]
            b = self.__weights['b' + str(le)]
            A_prev = self.__cache['A' + str(le - 1)]

            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)
            self.__cache['A' + str(le)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression

        Args:
            Y (array): is a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
            A (array): is a numpy.ndarray with shape (1, m)
            containing the activated output of the neuron
            for each example
        """

        # number of examples
        m = Y.shape[1]

        # Compute cost using logistic regression
        cost = -(1 / m) * np.sum(
          Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
