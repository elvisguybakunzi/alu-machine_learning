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
