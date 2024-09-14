#!/usr/bin/env python3
"""This Script converts a one-hot matrix
into a vector of labels"""


import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into
    a vector of labels

    Args:
        one_hot (numpy.ndarray): A one-hot encoded array
        with shape (classes, m)

    Returns:
        numpy.ndarray: A vector of labels with shape (m,)
        or None if input is invalid
    """

    try:
        # Validate input
        if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
            return None

        # Decode by finding the index of the maximum value in each column
        labels = np.argmax(one_hot, axis=0)

        return labels
    except Exception:
        return None
