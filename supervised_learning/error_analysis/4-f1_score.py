#!/usr/bin/env python3
"""Calculate the F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculate the F1 score for each class in a confusion matrix.

    Args:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                               where row indices represent the correct labels
                               and column indices represent
                               the predicted labels.

    Returns:
    numpy.ndarray: Array of shape (classes,) containing the F1 score
                   of each class.
    """
    # Calculate sensitivity (recall) and precision
    sens = sensitivity(confusion)
    prec = precision(confusion)

    # Calculate F1 score
    f1 = 2 * (prec * sens) / (prec + sens)

    # Handle division by zero
    f1 = np.nan_to_num(f1, nan=0.0)

    return f1
