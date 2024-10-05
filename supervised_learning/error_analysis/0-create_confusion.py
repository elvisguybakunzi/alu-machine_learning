#!/usr/bin/env python3
"""creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix

    Args:
        labels (numpy.ndarray): One-hot encoded true labels
        of shape (m, classes)
        logits (numpy.ndarray): One-hot encoded predicted labels
        of shape (m, classes)

    Return:
        numpy.ndarray: Confusion matrix of shape (classes, classes)
    """
    # Get the numbers of classes
    classes = labels.shape[1]

    # Convert one-hot encoded arrays to class indices
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    # Create the confusion matrix
    confusion = np.zeros((classes, classes), dtype=np.float64)

    # Populate the confusion matrix
    for true, pred in zip(true_classes, predicted_classes):
        confusion[true][pred] += 1
    return confusion
