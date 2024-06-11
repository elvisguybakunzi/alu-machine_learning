#!/usr/bin/env python3
'''This script deals with multivariate distribution'''
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes a Multivariate Normal distribution.
        Parameters:
        data (numpy.ndarray): The data set of shape (d, n)
        Raises:
        TypeError: If data is not a 2D numpy.ndarray
        ValueError: If n is less than 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)

    @property
    def mean(self):
        """
        Returns the mean of the data set.
        """
        return self._mean

    @mean.setter
    def mean(self, value):
        self._mean = value

    @property
    def cov(self):
        """
        Returns the covariance matrix of the data set.
        """
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = value
