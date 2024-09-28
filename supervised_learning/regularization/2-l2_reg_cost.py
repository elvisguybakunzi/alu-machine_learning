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
    reg_loss = tf.losses.get_regularization_losses()
    total_loss = cost + reg_loss
    return total_loss
