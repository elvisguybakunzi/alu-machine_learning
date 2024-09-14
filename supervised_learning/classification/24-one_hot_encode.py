#!/usr/bin/env python3
"""This Script converts a numeric label vector
into a one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into
    a one-hot matrix

    Args:
        Y (numpy.ndarray): A vector of numeric class labels
        with shape (m,)
        classes (int): The maximum number of classes found in Y

    Returns:
      numpy.ndarray: One-hot encoding of Y with
      shape (classes, m)
      or None if input is invalid
    """

    try:
        # Validate input
        if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
            return None
        if len(Y.shape) != 1 or classes <= np.max(Y):
            return None

        # Create a one-hot matrix
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))

        # set the appropriate indices to 1
        one_hot[Y, np.arange(m)] = 1

        return one_hot
    except Exception:
        return None
