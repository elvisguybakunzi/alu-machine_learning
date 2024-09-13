#!/usr/bin/env python3
"""Script that defines a neural
network with one hidden layer
"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one
    hidden layer with binary
    classification
    """
    def __init__(self, nx, nodes):
        """Initialize the neural network

        Args:
            nx (int): is the number of input
            features

            nodes (int): is the number of nodes
            found in the hidden layer
        Raises:
            TypeError: If nx is not an integer or
            nodes is not an integer

            ValueError: If nx is less than 1 or
            nodes is less than 1
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes ,ust be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private attributes Hidden layers
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private attributes Output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter methods of each private attributes
    @property
    def W1(self):
        """Getter of W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter of b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter of A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter of W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter of b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter of A2"""
        return self.__A2

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """Calculates the forward
        propagation of the neural network

        Args:
            X (array):  is a numpy.ndarray
            with shape (nx, m) that contains
            the input data
        """

        # The hidden layer
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)

        # The output layer
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the
        model using logistic regression

        Args:
            Y (array):  is a numpy.ndarray
            with shape (1, m) that contains
            the correct labels for the input data

            A (array): _description_is a numpy.ndarray
            with shape (1, m) containing the activated
            output of the neuron for each example
        """

        m = Y.shape[1]

        # Calculate the cost
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
