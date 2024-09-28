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
    # Define the L2 regularizer
    l2_regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    
    # Create the layer using Dense, specifying the L2 regularizer for the kernel
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
                            kernel_regularizer=l2_regularizer)
    
    # Return the output of the new layer applied
    # to the previous layer's output
    return layer(prev)
