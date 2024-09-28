#!/usr/bin/env python3
"""Create layer"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    Arguments:
    - prev: tensor containing the output of the previous layer
    - n: number of nodes the new layer should contain
    - activation: activation function that should be used on the layer
    - lambtha: L2 regularization parameter

    Returns:
    - The output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    l2_loss = tf.contrib.layers.l2_regularizer(scale=lambtha)
    hidden_layer = tf.layers.Dense(units=n, activation=activation,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=l2_loss)
    output = hidden_layer(prev)
    return output
