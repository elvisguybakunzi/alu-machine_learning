#!/usr/bin/env python3
"""This Script calculates the derivatie of
    of a polynamial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
    poly (list): A list of coefficients representing a polynomial.

    Returns:
    list: A new list of coefficients representing
    the derivative of the polynomial.
    None: If poly is not valid.
    """
    if not isinstance(poly, list):
        return None

    if len(poly) < 2:
        return None

    elif len(poly) <= 1:
        return [0]

    result = []

    for i in range(1, len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None

        new_coeff = i * poly[i]
        result.append(new_coeff)

    if not result:
        return [0]

    return result
