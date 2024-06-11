#!/usr/bin/env python3
'''This script calculates the correlation matrix'''
import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.
    Parameters:
    C (numpy.ndarray): The covariance matrix of shape (d, d)
    Returns:
    numpy.ndarray: The correlation matrix of shape (d, d)
    Raises:
    TypeError: If C is not a numpy.ndarray
    ValueError: If C does not have shape (d, d)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d = C.shape[0]
    stddev = np.sqrt(np.diag(C))
    if np.any(stddev == 0):
        raise ValueError("Covariance matrix contains zero variance elements")

    corr = C / np.outer(stddev, stddev)
    return corr
