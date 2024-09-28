#!/usr/bin/env python3

import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    # Initialize the layer weights using He et al. method
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    
    # Create the layer
    layer = tf.layers.dense(inputs=prev, 
                            units=n, 
                            activation=None,
                            kernel_initializer=initializer)
    
    # Apply the activation function
    if activation is not None:
        layer = activation(layer)
    
    # Apply dropout
    dropout = tf.nn.dropout(layer, keep_prob)
    
    return dropout