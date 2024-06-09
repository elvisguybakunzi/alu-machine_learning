#!/usr/bin/env python3
"""
Binomial module
"""

class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate p from data
            p = sum(data) / (len(data) * n)
            # Round n to the nearest integer
            self.n = round(n)
            # Recalculate p
            self.p = sum(data) / (len(data) * self.n)
