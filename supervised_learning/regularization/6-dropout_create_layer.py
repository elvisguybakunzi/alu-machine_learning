#!/usr/bin/env python3

import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    # He initialization for weights
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    
    # Define the dense layer with the specified number of nodes
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=initializer)(prev)
    
    # Apply dropout
    dropout_layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    
    return dropout_layer