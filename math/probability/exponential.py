#!/usr/bin/env python3
"""
Exponential module
"""


class Exponential:
    """Represents an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha as the inverse of the mean of the data
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculate the PDF for a given time period"""
        if x < 0:
            return 0
        # Calculate e^(-lambtha * x)
        e = 2.7182818285
        e_lambtha_x = e ** (-self.lambtha * x)
        # Calculate PDF
        pdf = self.lambtha * e_lambtha_x
        return pdf

    def cdf(self, x):
        """Calculate the CDF for a given time period"""
        if x < 0:
            return 0
        # Calculate 1 - e^(-lambtha * x)
        e = 2.7182818285
        e_lambtha_x = e ** (-self.lambtha * x)
        # Calculate CDF
        cdf = 1 - e_lambtha_x
        return cdf
