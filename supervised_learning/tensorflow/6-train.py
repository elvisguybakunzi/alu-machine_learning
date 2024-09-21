#!/usr/bin/env python3
"""Training the model"""

import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Parameters:
    - X_train: numpy.ndarray of shape (m, n), containing the training data
    - Y_train: numpy.ndarray of shape (m, c), containing the training labels
    - X_valid: numpy.ndarray of shape (m_v, n), containing the validation data
    - Y_valid: numpy.ndarray of shape (m_v, c), containing
    the validation labels
    - layer_sizes: list, number of nodes in each layer of the network
    - activations: list, activation functions for each layer
    - alpha: float, learning rate
    - iterations: int, number of iterations to train over
    - save_path: string, path to save the model

    Returns:
    - The path where the model was saved.
    """

    # Import necessary functions
    a = __import__('0-create_placeholders').create_placeholders
    create_placeholders = a
    forward_prop = __import__('2-forward_prop').forward_prop
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_train_op = __import__('5-create_train_op').create_train_op

    # Create placeholders for input data
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation to predict output
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate the loss
    loss = calculate_loss(y, y_pred)

    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Add the required tensors and operations to the collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Create a saver object to save the model
    saver = tf.train.Saver()

    # Start a TensorFlow session to train the model
    with tf.Session() as sess:
        sess.run(init)  # Initialize variables

        # Training loop
        for i in range(iterations + 1):
            # Train the model using the training data
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                # Calculate training cost and accuracy
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_train, y: Y_train})

                # Calculate validation cost and accuracy
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_valid, y: Y_valid})

                # Print the training and validation metrics
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model after training is complete
        save_path = saver.save(sess, save_path)

    return save_path
