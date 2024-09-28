#!/usr/bin/env python3
'''function that builds, trains, and saves a neural network model
  in tensorflow using adam optimization, mini_gradient descent,
 learning rate decay, and batch normalization'''
import numpy as np
import tensorflow as tf

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in TensorFlow."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    
    # Initialize placeholders for input and labels
    nx = X_train.shape[1]
    ny = Y_train.shape[1]
    
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, ny], name='y')
    
    # Learning rate placeholder
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate)
    
    # Initialize the model
    def forward_prop(x):
        for i in range(len(layers)):
            if i == 0:
                layer = tf.layers.dense(x, units=layers[i], activation=None,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            else:
                layer = tf.layers.dense(layer, units=layers[i], activation=None,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            # Batch normalization
            layer = tf.layers.batch_normalization(layer)
            # Apply activation function
            if activations[i]:
                layer = activations[i](layer)
        return layer
    
    # Build the graph
    output = forward_prop(x)
    
    # Define the loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
    
    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Define Adam optimizer with momentum (beta1, beta2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    # Initialize all variables
    init = tf.global_variables_initializer()
    
    # Saver to save the model
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        num_batches = X_train.shape[0] // batch_size
        if X_train.shape[0] % batch_size != 0:
            num_batches += 1
        
        # Training loop
        for epoch in range(epochs):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            
            print("After {} epochs:".format(epoch))
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                
                if (batch + 1) % 100 == 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(batch + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        
        # Save the model
        save_path = saver.save(sess, save_path)
        print("Model saved in path: {}".format(save_path))
        
    return save_path

def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way."""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
