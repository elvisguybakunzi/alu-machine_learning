#!/usr/bin/env python3
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

    def recursive_sum(k):
        if k == 0:
            return 0
        return k**2 + recursive_sum(k - 1)

    return recursive_sum(n)
