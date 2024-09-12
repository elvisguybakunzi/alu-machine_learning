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
