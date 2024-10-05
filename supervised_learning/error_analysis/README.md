# Error Analysis in Machine Learning

These are concepts in Machine Learning used in Error Analysis:


### Confusion Matrix:

A confusion matrix is a table used to describe the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.

### Type I and Type II errors:


- Type I error (False Positive): Rejecting a null hypothesis when it's actually true.

- Type II error (False Negative): Failing to reject a null hypothesis when it's actually false.


### Sensitivity, Specificity, Precision, and Recall:


- Sensitivity (True Positive Rate or Recall): The proportion of actual positive cases correctly identified.
- Specificity (True Negative Rate): The proportion of actual negative cases correctly identified.
- Precision: The proportion of predicted positive cases that are actually positive.
- Recall: Same as sensitivity.


### F1 Score:

The F1 score is the harmonic mean of precision and recall, providing a single score that balances both metrics.

### Bias and Variance:


- Bias: The error from erroneous assumptions in the learning algorithm.

- Variance: The error from sensitivity to small fluctuations in the training set.


### Irreducible Error:

The error that cannot be reduced by any model due to the inherent noise in the data.

### Bayes Error:

The lowest possible error rate for any classifier on a given problem and is the irreducible error.

### Approximating Bayes Error:

This can be done by using very complex models or ensemble methods and assuming their error rate approaches the Bayes error.

### Calculating Bias and Variance:

This involves comparing the model's predictions to the true values across multiple training sets.

### Creating a Confusion Matrix:

This is typically done by comparing the predicted labels to the true labels for a set of examples.