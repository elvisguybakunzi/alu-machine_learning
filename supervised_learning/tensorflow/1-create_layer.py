#!/usr/bin/env python3
"""Creation of layer"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Parameters:
    prev: tf.Tensor
        The tensor output from the previous layer.
    n: int
        The number of nodes in the layer to create.
    activation: function
        The activation function to use on the layer.

    Returns:
    tf.Tensor
        The tensor output of the layer.
    """
    # He initialization for the layer weights
    a = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    initializer = a

    # Creating the layer with the given activation function
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')

    return layer(prev)
