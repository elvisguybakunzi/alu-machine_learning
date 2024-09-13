#!/usr/bin/env python3
"""Script that defines a neural
network with one hidden layer
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Defines a neural network with one
    hidden layer with binary
    classification
    """
    def __init__(self, nx, nodes):
        """Initialize the neural network

        Args:
            nx (int): is the number of input
            features

            nodes (int): is the number of nodes
            found in the hidden layer
        Raises:
            TypeError: If nx is not an integer or
            nodes is not an integer

            ValueError: If nx is less than 1 or
            nodes is less than 1
        """

        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes ,ust be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private attributes Hidden layers
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private attributes Output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter methods of each private attributes
    @property
    def W1(self):
        """Getter of W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter of b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter of A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter of W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter of b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter of A2"""
        return self.__A2

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """Calculates the
        propagation of the neural network

        Args:
            X (array):  is a numpy.ndarray
            with shape (nx, m) that contains
            the input data
        """

        # The hidden layer
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)

        # The output layer
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the
        model using logistic regression

        Args:
            Y (array):  is a numpy.ndarray
            with shape (1, m) that contains
            the correct labels of the input data

            A (array): _description_is a numpy.ndarray
            with shape (1, m) containing the activated
            output of the neuron of each example
        """

        m = Y.shape[1]

        # Calculate the cost
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network
        predictions

        Args:
            X (array): is a numpy.ndarray with
            shape (nx, m) that contains the input data

            Y (array): is a numpy.ndarray with shape (1, m)
            that contains the correct labels of
            the input data
        """

        # Calculate propagation
        A1, A2 = self.forward_prop(X)

        # Convert probabilities A2 to binary predictions
        prediction = np.where(A2 >= 0.5, 1, 0)

        # Calculate the cost
        cost = self.cost(Y, A2)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent
        on the neural network

        Args:
            X (array): is a numpy.ndarray with
            shape (nx, m) that contains the input data

            Y (array): is a numpy.ndarray with
            shape (1, m) that contains the
            correct labels of the input data

            A1: is the output of the hidden layer

            A2: is the predicted output

            alpha (float, optional): is the learning rate
            Defaults to 0.05.
        """

        m = X.shape[1]

        # Calculate the gradient of W2 and b2
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # Calculate the gradient of W1 and b1
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update the weights and biases
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network

        Args:
            X (array): is a numpy.ndarray with
            shape (nx, m) that contains the input data

            Y (array): is a numpy.ndarray with
            shape (1, m) that contains the correct
            labels of the input data

            iterations (int, optional): is the number of iterations
            to train over
            Defaults to 5000.

            alpha (float, optional): is the learning rate
            Defaults to 0.05.

            verbose (bool): whether to print the cost every
            step iteration

            graph (bool): whether to plot the cost over training

            step (int): frequency of printing or plotting
        """
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
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
          
        costs = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            
            if i % step == 0:
                cost = self.cost(Y, A2)
                costs.append(cost)
                if verbose:
                    print("cost after {} iterations: {}".format(i, cost))
        if graph:
            plt.plot(range(0, iterations +1, step), costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
