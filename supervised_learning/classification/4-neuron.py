#!/usr/bin/env python3
"""This Script defines a single neuron."""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """
        Initializes a neuron for binary classification.

        Args:
            nx (integer): The number odf input features to the neuron.
        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter function for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter function for the Activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        This function calculates the forward propagation of neuron using
        sigmoid activation function.

        Args:
            X (array): A numpy.ndarray with shape (nx, m) containing
            the input data, where nx is the number of input features,
            and m is the number of examples.

        Returns:
            The updated private attributes __A (the activated output)
        """
        # Linear transformation
        Z = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        This Public method calculates the cost of the model using
        logistic regression

        Args:
            Y (array): numpy.ndarray with shape (1, m) that contains
            correct labels for the input data.
            A (array): numpy.ndarray with shape (1, m) containing
            the activated output of the neuron of each example

        Returns:
            The cost of the model as a float
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

        return cost

    def evaluate(self, X, Y):
        """_summary_

        Args:
            X (array): is a numpy.ndarray with shape
            (nx, m) that contains the input data
            Y (array):  is a numpy.ndarray with shape (1, m)
            that contains the correct labels for
            the input data
        Returns:
            The neuron's prediction (numpy.ndarray) and the
            cost the network (float)
        """
        # Forward propagation to calculate A
        A = self.forward_prop(X)

        # Prediction is 1 if A >= 0.5, else 0
        prediction = np.where(A >= 0.5, 1, 0)

        # Calculate the cost
        cost = self.cost(Y, A)

        return prediction, cost
