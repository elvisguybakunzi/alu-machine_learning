#!/usr/bin/env python3
"""Training operation"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network
    using gradient descent.

    Parameters:
    loss: tf.Tensor
        The loss of the network's prediction.
    alpha: float
        The learning rate for gradient descent.

    Returns:
    tf.Operation
        An operation that trains the network using
        gradient descent.
    """
    # Create the gradient descent optimizer with the specified learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Use the optimizer to minimize the loss
    # (this creates the training operation)
    train_op = optimizer.minimize(loss)

    return train_op
