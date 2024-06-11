#!/usr/bin/env python3
'''The Script that calculates the mean and covariance'''
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.
    Parameters:
    X (numpy.ndarray): The data set of shape (n, d)
    Returns:
    mean (numpy.ndarray): The mean of the data set of shape (1, d)
    cov (numpy.ndarray): The covariance matrix of the data set of shape (d, d)
    Raises:
    TypeError: If X is not a 2D numpy.ndarray
    ValueError: If n is less than 2
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)
    return mean, cov
