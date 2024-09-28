#!/usr/bin/env python3
"""Cost with Regularization"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Arguments:
    - cost: tensor containing the cost of the network
    without L2 regularization.

    Returns:
    - tensor containing the cost of the network accounting
    for L2 regularization.
    """
    # Get the L2 regularization losses
    # from the collection of regularization losses
    l2_regularization_losses = tf.losses.get_regularization_losses()

    # Add the L2 regularization losses to the original cost
    return cost + tf.reduce_sum(l2_regularization_losses)
