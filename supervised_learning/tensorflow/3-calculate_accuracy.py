#!/usr/bin/env python3
"""Calculate the accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Parameters:
    y: tf.placeholder
        Placeholder for the true labels of the input data
        (one-hot encoded).
    y_pred: tf.Tensor
        Tensor containing the network's predictions (raw logits).

    Returns:
    tf.Tensor
        A tensor containing the decimal accuracy of the prediction.
    """
    # Get the index of the highest predicted probability
    # for each class (argmax of predictions)
    correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))

    # Cast the boolean values to float32
    # (True -> 1.0, False -> 0.0) and calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy
