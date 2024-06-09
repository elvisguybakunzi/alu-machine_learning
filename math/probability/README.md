# Probability Concepts in Python

This project provides an overview of fundamental probability concepts and demonstrates how to implement them in Python. By the end of this project, you will be able to explain basic probability concepts and write Python code to calculate probabilities, mean, variance, and standard deviation.

## Learning Objectives

At the end of this project, you should be able to explain the following concepts to anyone:

- What is probability?
- Basic probability notation
- What is independence? What is disjoint?
- What is a union? What is an intersection?
- What are the general addition and multiplication rules?
- What is a probability distribution?
- What is a probability distribution function? What is a probability mass function?
- What is a cumulative distribution function?
- What is a percentile?
- What is mean, standard deviation, and variance?
- Common probability distributions

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- All files should end with a new line
- The first line of all files should be exactly `#!/usr/bin/env python3`
- A `README.md` file at the root of the project folder is mandatory
- Code should use the `pycodestyle` style (version 2.5)
- All modules should have documentation
- All classes should have documentation
- All functions (inside and outside a class) should have documentation
- Unless otherwise noted, importing modules is not allowed
- All files must be executable
- The length of files will be tested using `wc`

## Concepts Explained

### Probability

Probability is the measure of the likelihood that an event will occur, quantified as a number between 0 and 1.

### Basic Probability Notation

- **P(A)**: Probability of event A occurring.
- **P(A ∩ B)**: Probability of both events A and B occurring (intersection).
- **P(A ∪ B)**: Probability of either event A or B occurring (union).

### Independence and Disjoint

- **Independent events**: Two events are independent if the occurrence of one does not affect the probability of the other.
- **Disjoint (mutually exclusive) events**: Two events are disjoint if they cannot both occur at the same time.

### Union and Intersection

- **Union (A ∪ B)**: The event that either A or B or both occur.
- **Intersection (A ∩ B)**: The event that both A and B occur.

### Addition and Multiplication Rules

- **Addition Rule**: For any two events A and B:
  - \( P(A \cup B) = P(A) + P(B) - P(A \cap B) \)
- **Multiplication Rule**: For any two events A and B:
  - If A and B are independent: \( P(A \cap B) = P(A) \times P(B) \)
  - Otherwise: \( P(A \cap B) = P(A) \times P(B|A) \)

### Probability Distribution

A probability distribution describes how the probabilities are distributed over the values of the random variable.

### Probability Distribution Function (PDF) and Probability Mass Function (PMF)

- **PDF**: For continuous variables, it describes the probability density.
- **PMF**: For discrete variables, it gives the probability that a discrete random variable is exactly equal to some value.

### Cumulative Distribution Function (CDF)

The CDF of a random variable X is the function F(x) that gives the probability that the variable takes on a value less than or equal to x.

### Percentile

A percentile indicates the value below which a given percentage of observations in a group falls.

### Mean, Standard Deviation, and Variance

- **Mean (μ)**: The average value.
- **Variance (σ²)**: The average of the squared differences from the mean.
- **Standard Deviation (σ)**: The square root of the variance.

### Common Probability Distributions

- **Binomial Distribution**: For a fixed number of trials, each with the same probability of success.
- **Normal Distribution**: Symmetric, bell-shaped distribution characterized by mean μ and standard deviation σ.
- **Poisson Distribution**: For the number of events occurring within a fixed interval of time or space.

## Python Implementation

The provided Python script demonstrates these concepts with functions to calculate probability, union, intersection, mean, variance, and standard deviation.

### Usage

Save the following code to a file named `probability.py` and make it executable:

```python
#!/usr/bin/env python3

import math

def probability(event_outcomes, sample_space):
    """Calculate the probability of an event."""
    return event_outcomes / sample_space

def union(prob_A, prob_B, prob_A_and_B):
    """Calculate the union of two probabilities."""
    return prob_A + prob_B - prob_A_and_B

def intersection(prob_A, prob_B):
    """Calculate the intersection of two independent events."""
    return prob_A * prob_B

def mean(data):
    """Calculate the mean of a dataset."""
    return sum(data) / len(data)

def variance(data):
    """Calculate the variance of a dataset."""
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)

def standard_deviation(data):
    """Calculate the standard deviation of a dataset."""
    return math.sqrt(variance(data))

# Example usage
if __name__ == "__main__":
    # Probability example
    prob_A = probability(1, 2)
    prob_B = probability(1, 4)
    print(f"P(A): {prob_A}")
    print(f"P(B): {prob_B}")

    # Union and intersection
    union_prob = union(prob_A, prob_B, probability(1, 8))
    intersection_prob = intersection(prob_A, prob_B)
    print(f"P(A ∪ B): {union_prob}")
    print(f"P(A ∩ B): {intersection_prob}")

    # Mean, variance, standard deviation
    data = [1, 2, 3, 4, 5]
    print(f"Mean: {mean(data)}")
    print(f"Variance: {variance(data)}")
    print(f"Standard Deviation: {standard_deviation(data)}")

