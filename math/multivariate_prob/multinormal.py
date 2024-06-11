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

    def pdf(self, x):
        """
        Calculates the PDF at a data point.
        Parameters:
        x (numpy.ndarray): The data point of shape (d, 1)
        Raises:
        TypeError: If x is not a numpy.ndarray
        ValueError: If x is not of shape (d, 1)
        Returns:
        float: The value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
