#!/usr/bin/env python3
"""Calculate the specificity"""

import numpy as np


def specificity(confusion):
    """
    Calculate the specificity for each class in a confusion matrix.

    Args:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                               where row indices represent the correct labels
                               and column indices represent
                               the predicted labels.

    Returns:
    numpy.ndarray: Array of shape (classes,) containing
                   the specificity of each class.
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (FP + FN + TP)

    TNR = TN / (TN + FP)
    return TNR
