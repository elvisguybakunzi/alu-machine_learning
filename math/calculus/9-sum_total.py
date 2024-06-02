#!/usr/bin/env python3
"""A Script that calculates a sigma notation"""


def summation_i_squared(n):
    """
    Calculate the sum of the squares of the first n natural numbers.

    Args:
    n (int): The stopping condition.

    Returns:
    int: The integer value of the sum if n is valid.
    None: If n is not a valid number.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    recursive_sum = n * (n + 1) * (2 * n + 1) // 6

    return recursive_sum
