#!/usr/bin/env python3
"""Forward Propagation"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Parameters:
    x: tf.placeholder
        The placeholder for the input data.
    layer_sizes: list of int
        A list containing the number of nodes
        in each layer of the network.
    activations: list of functions
        A list containing the activation functions
        for each layer of the network.

    Returns:
    tf.Tensor
        The prediction of the network in tensor form.
    """
    layer = x  # Start with the input layer
    for i in range(len(layer_sizes)):
        # Apply each layer with the respective size and activation function
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
