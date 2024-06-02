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
    list: A new list of coefficients representing the derivative of the polynomial.
    None: If poly is not valid.
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    if len(poly) <= 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative

if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))
