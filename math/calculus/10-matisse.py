#!/usr/bin/env python3
'''The Script that calculates the derivative of a polynomial'''


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
    poly (list): A list of coefficients representing a polynomial.

    Returns:
    list: A new list of coefficients representing the
    derivative of the polynomial.
    None: If poly is not valid.
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float))
                                             for coef in poly):
        return None

    if len(poly) == 0:
        return None

    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative if derivative else [0]
