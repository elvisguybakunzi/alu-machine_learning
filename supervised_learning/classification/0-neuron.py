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

        # initialize weights using random normal distribution
        self.W = np.random.randn(1, nx)

        # initialize bias to 0
        self.b = 0

        # initialize activated output (predicted) to 0
        self.A = 0
