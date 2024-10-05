#!/usr/bin/env python3
"""Calculate the precision"""

import numpy as np


def precision(confusion):
    """
    Calculate the precision for each class in a confusion matrix.

    Args:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                               where row indices represent the correct labels
                               and column indices represent
                               the predicted labels.

    Returns:
    numpy.ndarray: Array of shape (classes,) containing
                   the precision of each class.
    """
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        precision[i] = confusion[i][i] / np.sum(confusion[:, i])
    return precision
