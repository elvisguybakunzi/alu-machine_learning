#!/usr/bin/env python3
"""Training the model"""

import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier.
    
    Parameters:
    - X_train: numpy.ndarray of shape (m, n), containing the training data
    - Y_train: numpy.ndarray of shape (m, c), containing the training labels
    - X_valid: numpy.ndarray of shape (m_v, n), containing the validation data
    - Y_valid: numpy.ndarray of shape (m_v, c), containing the validation labels
    - layer_sizes: list, number of nodes in each layer of the network
    - activations: list, activation functions for each layer
    - alpha: float, learning rate
    - iterations: int, number of iterations to train over
    - save_path: string, path to save the model
    
    Returns:
    - The path where the model was saved.
    """
    # Import required functions
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    create_train_op = __import__('5-create_train_op').create_train_op
    forward_prop = __import__('2-forward_prop').forward_prop

    # Get the number of features and classes
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    # Create placeholders
    x, y = create_placeholders(nx, classes)

    # Create the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add placeholders and tensors to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create saver
    saver = tf.train.Saver()

    # Start session
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Calculate training cost and accuracy
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            
            # Calculate validation cost and accuracy
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            # Print stats every 100 iterations, at 0th iteration, and at last iteration
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            # Perform training step if not at last iteration
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
