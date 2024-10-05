#!/usr/bin/env python3
"""Calculate the sensitivity"""

import numpy as np


def sensitivity(confusion):
    """
    Calculate the sensitivity for each class in a confusion matrix.

    Args:
    confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
                               where row indices represent the correct labels
                               and column indices represent
                               the predicted labels.

    Returns:
    numpy.ndarray: Array of shape (classes,) containing the sensitivity
                    of each class.
    """
    # Get the number of classes
    classes = confusion.shape[0]

    # Initialize an array to store sensitivities
    sensitivities = np.zeros(classes)

    # Calculate sensitivity for each class
    for i in range(classes):
        # True positives are on the diagonal of the confusion matrix
        true_positives = confusion[i, i]
        # Sum of the row gives all actual positives for this class
        actual_positives = np.sum(confusion[i, :])

        # Sensitivity = True Positives / Actual Positives
        # Handle division by zero
        if actual_positives != 0:
            sensitivities[i] = true_positives / actual_positives
        else:
            sensitivities[i] = 0

    return sensitivities
