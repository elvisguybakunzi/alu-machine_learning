#!/usr/bin/env python3
"""Create layer"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    hidden_layer = tf.layers.Dense(units=n, activation=activation,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=dropout)
    output = hidden_layer(prev)
    return output
