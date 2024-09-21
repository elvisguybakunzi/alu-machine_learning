#!/usr/bin/env python3
"""Evaluate the output"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
    - X: numpy.ndarray containing the input data to evaluate
    - Y: numpy.ndarray containing the one-hot labels for X
    - save_path: string, the location to load the model from

    Returns:
    - The network’s prediction, accuracy, and loss, respectively
    """

    # Start a new TensorFlow session
    with tf.Session() as sess:
        # Load the saved model's meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')

        # Restore the weights from the checkpoint
        saver.restore(sess, save_path)

        # Access the default graph
        graph = tf.get_default_graph()

        # Retrieve tensors from the collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Create a feed dictionary with the input data
        feed_dict = {x: X, y: Y}

        # Evaluate predictions, accuracy, and loss
        predictions = sess.run(y_pred, feed_dict=feed_dict)
        acc = sess.run(accuracy, feed_dict=feed_dict)
        cost = sess.run(loss, feed_dict=feed_dict)

    return predictions, acc, cost
