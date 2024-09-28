# Regularization in Machine Learning

Regularization is a technique used to prevent overfitting in machine learning models by adding a penalty to the loss function. This penalty discourages the model from fitting too closely to the training data, ensuring that the model generalizes well to unseen data.

## Purpose of Regularization

The main purpose is to improve the generalization of a model by preventing it from overfitting to the noise in the training data.
It helps in controlling model complexity by penalizing large weights or complex models.

### Types of Regularization

#### 1. L1 Regularization (Lasso)

*Definition*: L1 regularization adds a penalty equal to the absolute value of the magnitude of coefficients.

*Effect*: It encourages sparsity in the weights, effectively setting some weights to zero, which can result in feature selection.

- Pros:
Encourages sparsity (can be used for feature selection).
Works well when some features are irrelevant.

- Cons:
Can be less stable than L2 when used with high-dimensional data.

#### 2. L2 Regularization (Ridge)

*Definition*: L2 regularization adds a penalty equal to the squared magnitude of the coefficients.

*Effect*: It discourages large weights but doesn’t set them to zero.

- Pros:
Helps prevent overfitting.
Works well with correlated features.

- Cons:
Does not perform feature selection as it doesn’t set weights to zero.

#### Difference between L1 and L2 Regularization

- L1 results in sparse solutions, meaning many weights will be zero, leading to simpler models.
- L2 results in weights that are small but non-zero, keeping all features but reducing their impact.

## Dropout

*Definition:* Dropout is a technique used during training where, at each iteration, a subset of neurons is randomly "dropped" (set to zero).

*Purpose:* Prevents neurons from becoming too dependent on each other, helping the model generalize better.

- Pros:
Prevents co-adaptation of neurons.
Simple and effective for preventing overfitting.

- Cons:
Adds randomness, which can slow down convergence.

## Early Stopping

*Definition:* Early stopping halts the training process when the validation error stops decreasing after several iterations.

*Purpose:* Prevents the model from overfitting by stopping training at the point of optimal performance on validation data.

- Pros:
Simple to implement.
Helps prevent overfitting.

- Cons:
Needs a validation set, which reduces training data.
Can halt too early, leading to underfitting.

## Data Augmentation

*Definition:* Data augmentation involves artificially increasing the size of the training dataset by applying random transformations (e.g., rotations, flips) to the training data.

*Purpose:* Helps to increase the diversity of the training set and improves model robustness.

- Pros:
Increases data diversity.
Improves generalization.

- Cons:
Increases computation time during training.
