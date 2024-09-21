#!/usr/bin/env python3
"""Calculate the loss"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Parameters:
    y: tf.placeholder
        Placeholder for the true labels of the
        input data (one-hot encoded).
    y_pred: tf.Tensor
        Tensor containing the network's predictions
        (raw logits).

    Returns:
    tf.Tensor
        A tensor containing the loss of the prediction.
    """
    # Use tf.losses.softmax_cross_entropy to compute
    # the softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
