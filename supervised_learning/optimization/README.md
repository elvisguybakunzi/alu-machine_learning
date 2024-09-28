# Optimization in Machine Learning


For This assignment on optimization in machine learning, here is a detailed explanation of the concepts and their implementation steps in Python with the required tools and libraries, all tailored to meet the project guidelines:

## 1. What is a hyperparameter?
Definition: A hyperparameter is a configuration that is set before the training of a machine learning model. It controls the learning process, such as learning rate, batch size, and the number of epochs. Unlike model parameters, hyperparameters are not learned from the data.

### Example of Hyperparameters:
- Learning rate
- Batch size
- Number of layers in a neural network

## 2. How and why do you normalize your input data?

Why Normalize? Normalization helps ensure that the model converges faster during training by keeping all input features within the same scale. Without normalization, features with larger ranges may dominate the learning process.

### How to Normalize:
Subtract the mean of each feature and divide by the standard deviation (Z-score normalization).


```python

    def normalize(X):
        """Normalize input data X."""
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
```

3. What is a saddle point?
Definition: A saddle point is a point on the loss surface where the gradient is zero, but it is not a local minimum or maximum. This can slow down the convergence of gradient descent algorithms.


## 4. What is stochastic gradient descent (SGD)?
Definition: Stochastic Gradient Descent (SGD) is an optimization algorithm where each training example's gradient is computed individually and used to update the model. This contrasts with batch gradient descent, which uses the entire dataset.
Implementation:

```python

    def sgd(X, y, weights, learning_rate=0.01):
        """Perform one iteration of SGD."""
        for i in range(len(y)):
            gradient = compute_gradient(X[i], y[i], weights)
            weights = weights - learning_rate * gradient
        return weights

```

## 5. What is mini-batch gradient descent?
Definition: Mini-batch Gradient Descent splits the training dataset into small batches and computes the gradient on each batch. It strikes a balance between SGD and batch gradient descent.

#### Implementation:

```python

    def mini_batch_gradient_descent(X, y, weights, batch_size, learning_rate=0.01):
        """Perform Mini-batch Gradient Descent."""
        num_batches = len(X) // batch_size
        for i in range(num_batches):
            X_batch = X[i * batch_size:(i + 1) * batch_size]
            y_batch = y[i * batch_size:(i + 1) * batch_size]
            gradient = compute_gradient(X_batch, y_batch, weights)
            weights = weights - learning_rate * gradient
        return weights

```

## 6. What is a moving average? How do you implement it?

Definition: A moving average smooths out the data by averaging a window of past values. It's commonly used in optimization to reduce the variance in gradient updates.

#### Implementation:

```python

    def moving_average(data, window_size):
        """Compute the moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
```

### 7. What is gradient descent with momentum? How do you implement it?

Definition: Gradient descent with momentum helps accelerate gradients by considering the previous gradients, making optimization faster, especially in areas with high curvature.

#### Implementation:

```python

    def gradient_descent_momentum(X, y, weights, learning_rate=0.01, momentum=0.9):
        """Gradient Descent with Momentum."""
        velocity = np.zeros_like(weights)
        gradient = compute_gradient(X, y, weights)
        velocity = momentum * velocity - learning_rate * gradient
        weights += velocity
        return weights
```

## 8. What is RMSProp? How do you implement it?
Definition: RMSProp (Root Mean Square Propagation) adapts the learning rate based on the moving average of squared gradients, which helps to balance the learning rate.

#### Implementation:

```python

    def rmsprop(X, y, weights, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """RMSProp optimization algorithm."""
        cache = np.zeros_like(weights)
        gradient = compute_gradient(X, y, weights)
        cache = beta * cache + (1 - beta) * gradient**2
        weights -= learning_rate * gradient / (np.sqrt(cache) + epsilon)
        return weights
```

## 9. What is Adam optimization? How do you implement it?

Definition: Adam (Adaptive Moment Estimation) combines the advantages of RMSProp and momentum by keeping track of both the moving average of the gradient and the squared gradient.

#### Implementation:

```python

    def adam(X, y, weights, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimization algorithm."""
        m, v = np.zeros_like(weights), np.zeros_like(weights)
        t = 0
        gradient = compute_gradient(X, y, weights)
        t += 1
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return weights
```

## 10. What is learning rate decay? How do you implement it?

Definition: Learning rate decay reduces the learning rate during training to improve convergence and avoid overshooting.

Implementation:

```python

    def learning_rate_decay(initial_lr, decay_rate, epoch):
        """Apply learning rate decay."""
        return initial_lr / (1 + decay_rate * epoch)
```

## 11. What is batch normalization? How do you implement it?

Definition: Batch normalization normalizes the activations of a layer for each mini-batch to improve training speed and stability.

#### Implementation:

```python

def batch_norm(X, gamma, beta, epsilon=1e-8):
    """Batch normalization implementation."""
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    return gamma * X_norm + beta
```