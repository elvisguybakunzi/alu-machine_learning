#!/usr/bin/env python3
"""
Poisson module
"""


class Poisson:
    """Represents a Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the PMF for a given number of successes"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        # Calculate e^(-lambtha)
        e = 2.7182818285
        e_lambtha = e ** (-self.lambtha)
        # Calculate lambtha^k
        lambtha_k = self.lambtha ** k
        # Calculate k!
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        # Calculate PMF
        pmf = e_lambtha * lambtha_k / k_factorial
        return pmf
