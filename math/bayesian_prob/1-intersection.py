#!/usr/bin/env python3
"""
Module to calculate the likelihood of obtaining the observed data given
various hypothetical probabilities of developing severe side effects.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining the observed data given
    various hypothetical probabilities of developing severe side effects.

    Parameters:
    - x: The number of patients that develop severe side effects (integer).
    - n: The total number of patients observed (positive integer).
    - P: 1D numpy.ndarray containing the various hypothetical probabilities.

    Returns:
    - A 1D numpy.ndarray containing the likelihood of obtaining the data,
    x and n, for each probability in P, respectively.
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    a = np.math.factorial(n)
    b = np.math.factorial(x)
    c = np.math.factorial(n - x)
    binom_coeff = a / (b * c)

    # Calculate the likelihood for each probability in P
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods


def intersection(x, n, P, Pr):
    """
    Calculate the intersection of obtaining the observed data with
    various hypothetical probabilities of developing severe side effects.

    Parameters:
    - x: The number of patients that develop severe side effects (integer).
    - n: The total number of patients observed (positive integer).
    - P: 1D numpy.ndarray containing the various hypothetical probabilities.
    - Pr: 1D numpy.ndarray containing the prior beliefs of P.

    Returns:
    - A 1D numpy.ndarray containing the intersection of obtaining x and n
      with each probability in P, respectively.
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood
    L = likelihood(x, n, P)

    # Calculate the intersection
    intersection_values = L * Pr

    return intersection_values
