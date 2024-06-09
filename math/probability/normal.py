#!/usr/bin/env python3
"""
Normal module
"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate mean and standard deviation from data
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculate the z-score of a given x-value"""
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """Calculate the x-value of a given z-score"""
        x = z * self.stddev + self.mean
        return x

    def pdf(self, x):
        '''calculates the pdf of a value x'''
        pi = 3.1415926536
        e = 2.7182818285
        coeffecient = 1/(self.stddev * ((2*pi)**0.5))
        exponent = -0.5 * (((x - self.mean)/self.stddev) ** 2)
        pdf = coeffecient * e**exponent
        return pdf

    def cdf(self, x):
        '''calculates the cdf of a value x'''
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        # estimate the erf using taylor series expansion
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        # calculate the cdf from the estimated erf
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf
