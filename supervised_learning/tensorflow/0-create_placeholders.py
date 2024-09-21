#!/usr/bin/env python3
"""To return two placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network.

    Parameters:
    nx: int
        The number of feature columns in our input data
        (number of input features).
    classes: int
        The number of classes in our classifier
        (output size for one-hot labels).

    Returns:
    x: tf.placeholder
        Placeholder for the input data to the neural network.
    y: tf.placeholder
        Placeholder for the one-hot labels of the input data.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
