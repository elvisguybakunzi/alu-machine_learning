#!/usr/bin/env python3
"""This script performs multiclass classification"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network
    with multiclass classification.
    """

    def __init__(self, nx, layers):
        """class constructor

        Args:
            nx (int): is the number of input features
            layers (list): is a list representing the number
            of nodes in each layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(
          map(lambda layer: isinstance(layer, int) and layer > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for le in range(1, self.L + 1):
            layer_size = layers[le - 1]
            input_size = nx if le == 1 else layers[le - 2]

            self.__weights['W' + str(le)] = np.random.randn(
              layer_size, input_size) * np.sqrt(2 / input_size)
            self.__weights['b' + str(le)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter of the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter of the weights"""
        return self.__weights

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        """Softmax activation function"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """Calculates the propagation
        of the neural network

        Args:
            X (array): is a numpy.ndarray
            with shape (nx, m) that contains
            the input data
        """

        # Input layer stored as A0
        self.__cache['A0'] = X

        for le in range(1, self.__L + 1):
            W = self.__weights['W' + str(le)]
            b = self.__weights['b' + str(le)]
            A_prev = self.__cache['A' + str(le - 1)]

            Z = np.dot(W, A_prev) + b

            # Use softmax for the output layer, sigmoid for hidden layers
            if le == self.__L:
                A = self.softmax(Z)
            else:
                A = self.sigmoid(Z)

            self.__cache['A' + str(le)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model
        using categorical cross-entropy

        Args:
            Y (array): is a numpy.ndarray with shape (classes, m)
            that contains the correct labels of the input data
            A (array): is a numpy.ndarray with shape (classes, m)
            containing the activated output of the neuron
            of each example
        """

        # number of examples
        m = Y.shape[1]

        # Compute cost using categorical cross-entropy
        cost = -(1 / m) * np.sum(Y * np.log(A + 1e-8))

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network
        predictions

        Args:
            X (array): is a numpy.ndarray with shape (nx, m)
            that contains the input data
            Y (array): is a numpy.ndarray with shape (classes, m)
            that contains the correct labels of the input data
        """

        # propagation to get the network output
        A, _ = self.forward_prop(X)

        # Prediction: class with highest probability
        prediction = np.argmax(A, axis=0)
        prediction_one_hot = np.eye(Y.shape[0])[prediction].T

        # Calculate the cost
        cost = self.cost(Y, A)

        return prediction_one_hot, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent
        on the neural network

        Args:
            Y (array): is a numpy.ndarray with shape (classes, m)
            that contains the correct labels of
            the input data
            cache (dict): is a dictionary containing
            all the intermediary values of the network
            alpha (float, optional): is the learning rate
            Defaults to 0.05.
        """

        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache['A' + str(layer - 1)]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if layer > 1:
                W = self.__weights['W' + str(layer)]
                dA = np.dot(W.T, dZ)
                dZ = dA * (A_prev * (1 - A_prev))

            self.__weights['W' + str(layer)] -= alpha * dW
            self.__weights['b' + str(layer)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0:
            raise ValueError("step must be positive")

        costs = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose and i % step == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(range(0, iterations, step), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename (str): The file to which the object should
            be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename (str): The file from which the object should
            be loaded

        Returns:
            DeepNeuralNetwork: The loaded object, or None if filename
            doesn't exist
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
